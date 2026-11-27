# Nemotron-Labs-Image: autoregressive text to image

NVIDIA's nvidia/Nemotron-Labs-Image-30B-A3B (`src/nemotron_image/`) draws an image as a sequence
of visual tokens written by a language model, which a ViT tokenizer then renders to pixels:

- the language model is Nemotron 3 Nano's 30B-A3B hybrid (52 blocks: Mamba-2, attention and
  128-expert MoE; [`src/nemotron.rs`](nemotron.md#the-language-model-nemotron-h)), its vocabulary
  extended from 131,072 text tokens to 327,680 rows: three framing tokens (`<|img_start|>`,
  `<|img_end|>`, `<|vis_row|>`, ids 131072-131074) and the 196,560 visual tokens `<|vis_i|>`
  (ids 131075 + i);
- the tokenizer (`Nemotron-Labs-Image-Tokenizer/`, transformers `model_type` `l24sq_vit`, 642M
  parameters, [`src/nemotron_image/vit.rs`](../src/nemotron_image/vit.rs)) maps each 16x16 patch
  to one of 196,560 fixed Leech-lattice vectors and back.

```sh
./target/release/oxinfer --model nvidia/Nemotron-Labs-Image-30B-A3B --quant q4 \
    --generate-image "A cinematic photograph of a red fox in a snowy pine forest at dawn." -o fox.png
./target/release/oxinfer --model wiz0floyd/Nemotron-Labs-Image-30B-A3B-GGUF:Q4_0 \
    -p "A vintage red sports car parked on a coastal road at sunset." --size 1184x880 --seed 7 -o car.png
./target/release/oxinfer --model wiz0floyd/Nemotron-Labs-Image-30B-A3B-GGUF:Q4_0 \
    -p "a lighthouse in a storm, oil painting" --num-images 4 -o lighthouse.png   # lighthouse-0.png .. -3.png
./target/release/oxinfer serve --model nemo=wiz0floyd/Nemotron-Labs-Image-30B-A3B-GGUF:Q4_0
curl -s localhost:8080/v1/images/generations -d '{"model": "nemo", "prompt": "a red fox", "size": "1024x1024"}'
```

## The protocol

As the model card's `infer.py`:

- **Prompt**: the chat template with thinking off, the card's system prompt ("Generate an image
  matching the requested resolution by detailing the color, quantity, text, shape, size, texture,
  spatial relationships of the objects and background.") and the user turn
  `Generate a {W}x{H} image.\n\n{prompt}`, then `<|im_start|>assistant\n<think></think>`. No BOS
  (the template's tokens are encoded without special tokens).
- **Reply**: a line break, then `<|img_start|>`, the size header in plain text
  (`height=1024 width=1024`), and for each of the `H / 16` rows `<|vis_row|>` followed by `W / 16`
  visual tokens, then `<|img_end|>`. A 1024x1024 image is 64 rows of 64: 4,176 tokens after the
  prompt.
- **Sampling**: temperature 0.9, top-p 1, no top-k, no classifier-free guidance.
- **Sizes**: any multiple of 16; the card recommends 1024x1024, 1184x880, 1248x832 and 880x1184.
  The model was trained up to 1024x1024 pixels in area and follows 1024-class sizes; smaller ones
  are followed poorly (asked for 256x256 with `--unconstrained` it starts a 2640-row image, and
  the grid of a 256x256 comes out as a blurred corner of a larger picture).

By default the block's structure is enforced (`structure: grid`): the line break, the opening
marker, the header for the size asked for and every row marker are fed rather than sampled, and
the visual tokens are sampled among the visual vocabulary only (the logits of text tokens are
ignored inside the grid). Every generation then yields an image of the size asked for, where the
reference samples everything and rejects a generation whose header or grid is off. `--unconstrained`
(`structure: free`) samples every token as the reference does, up to `<|img_end|>`, and decodes the
grid the model wrote at the size its header announces (rows cut or padded to the header's width).

**Source images** (`--image FILE`, the server's image edits, `images_json` in the C API) are
encoded with the tokenizer's encoder (resized to at most 1024x1024 pixels in area, sides rounded
down to multiples of 16) into the same block, and by default given as earlier turns of the
conversation, then the instruction guided against the same prompt without it; the output keeps
the first source's size unless one is asked for. See [Editing](#editing) for what works and
what does not.

The model does not describe images: it was trained to write visual tokens, and was not tested
here as a text model.

## Editing

The model was trained with 10% text-and-image-to-image data ("TI2I": editing, generation from
reference images, multi-turn generation) in its three mid-training stages and none in its
reinforcement-learning stage; the paper (section 5.2, table 4) reports it as "included in
training but not evaluated". The release documents no prompt layout for it: the model card,
`infer.py`, the paper (`paper.pdf` in the repository), the chat template (no image content
type), `tokenization_unitron_lazy_vision.py` (it only maps `<|vis_i|>` / framing tokens to ids),
`generation_config.json` (empty) and the Hugging Face demo Space are all text to image; the
harness and VisionCode code are "releasing upon legal clearance".

oxinfer's layout and sampling for sources (the defaults; `source_layout`, `source_prompt`,
`guidance`, `image_guidance` change them):

```text
<|im_start|>system\n{system}<|im_end|>\n
<|im_start|>user\nGenerate a {w}x{h} image.\n\n{source_prompt}<|im_end|>\n      (per source)
<|im_start|>assistant\n<think></think>\n<|img_start|>height={h} width={w}<|vis_row|>...<|img_end|><|im_end|>\n
<|im_start|>user\nGenerate a {W}x{H} image.\n\n{instruction}<|im_end|>\n<|im_start|>assistant\n<think></think>
```

(the chat template's rendering of a past reply; `source_layout=inline` puts the blocks at the
start of the user turn instead), and classifier-free guidance on the instruction: logits
`u + 2 (c - u)`, `u` from the same prompt with an empty instruction, which shares the prompt's
prefix (one prefill of it, then two sequences, which take each token together in one GPU step
that reads the weights once: 101 tok/s against 141-147 for one sequence on the M5 Max).

What was checked (Q4_0 GGUF, M5 Max), and why edits are still not clean:

- The encoder is right: oxinfer's codes equal the reference's (above), and encoding then
  decoding a generated 1184x880 image gives it back with no visible loss.
- The model reads the sources: teacher-forced with the source's own codes as the reply
  ("Keep the image exactly the same."), it predicts 64% of them as its top choice (mean
  log-probability -2.8 over the visual vocabulary, entropy 4.5 nats; without the source 6%,
  -7.4, 7.9 nats). The inline layout scores -2.80, the turn layout -3.16 to -3.25, about
  equal; a source block without its `height=H width=W` header drops to -5.2, and a source
  encoded at half or 0.7 times the size to -7.7 / -8.0 (worse than none: the model copies by
  position, so a source of another size misleads it).
- The decoding is right: the GPU decode plan agrees with a one-pass CPU prefill at 8,342 tokens
  (largest logit difference 0.37, same top token), and the same edits sampled on the CPU come
  out the same way.
- Yet sampled at the card's temperature 0.9 without guidance, even "Keep the image exactly the
  same." returns a darker, foggy copy, and edits come out washed out (darker or whiter, low
  contrast, smeared). The visual vocabulary is 196,560 unit vectors of a Leech-lattice code
  whose nearest neighbours are 51 degrees apart (cosine 0.625, 128 of them per code): a code
  sampled off the source's is a large step, not a small one, and at a 36% miss rate the
  decoder averages them into haze. Text to image does not show this because reinforcement
  learning concentrated that distribution; the TI2I one was never tuned.
- Sharpening by truncation does not help: temperature 0.7 or 0.5, top-k 20 / 100 and top-p
  0.7 / 0.9 give horizontal streaks and repeated rows (the model copies its own last row
  instead of the source). Guidance does: at 2 the car of the test image comes out bright
  blue with its pose, the cliff and the road kept, and "add a golden retriever sitting on the
  road in front of the car" adds a dog; 3 follows the instruction more and the source less;
  image guidance (`image_guidance`, InstructPix2Pix's second scale, a third sequence) did not
  improve on it.

What remains: tones drift (the sunset sky of the car image comes out dark, the snowy fox scene
whiter and foggy), fine detail of the source is redrawn rather than kept, style changes
("turn this photo into a watercolor painting") are not followed, and results vary a lot between
seeds; generate several (`--num-images`) and pick. These are properties of the released
weights' untuned TI2I behaviour as far as the checks above can tell, not of the prompt layout
(no layout tried changed them: before or after the instruction, with or without the
`Generate a WxH image.` line or the system prompt, the source's own description in its turn).

## The tokenizer

Both halves are 24 Llama-style blocks of 1024 (16 heads of 64): RMSNorm (eps 1e-6),
bidirectional attention with 2D RoPE (the first half of each head rotated by the patch's row, the
second by its column, each half split in two as rotate-half RoPE does, base 10,000), RMSNorm and a
SwiGLU MLP of 2752. The encoder: a 16x16 stride-16 convolution of the pixels (in `[-1, 1]`,
padded by replication), the blocks, `quant_embed` to 24 values, the nearest codebook vector by
cosine. The decoder: the codebook rows (L2-normalized), `post_quant_embed` to 1024, the blocks,
`ffn` (1024 -> 3072, tanh), `conv_out` (3072 -> 3 x 16 x 16) and the patches laid out row by row,
clamped to `[-1, 1]`; pixels are `round((x + 1) / 2 * 255)`. The weights are float32; oxinfer
keeps them in bf16 (`vit_precision=f32` keeps float32), as the card runs the decoder.

GGUF conversions carry the language model only: the tokenizer folder is then fetched from
`nvidia/Nemotron-Labs-Image-30B-A3B` (2.6 GB, its other files skipped; `tokenizer=PATH|REPO`
points elsewhere).

## Options

| CLI | library (`GenerateOptions` / `LoadOptions`), C API and server keys | default |
|---|---|---|
| `--size WxH` | `width`, `height` (the server's `size`; 0 x 0: the first source's size) | 1024x1024; with source images, the first one's size |
| `--temp`, `--top-p`, `--top-k` | `temperature`, `top_p`, `top_k` | 0.9, 1, 0 |
| `--seed`, `--num-images N` | `seed`, `num_images` (the server's `n`; seeds `seed`, `seed + 1`, ...) | 42, 1 |
| `--system TEXT`, `--no-system` | `system_prompt` | the card's |
| `--unconstrained` | `structure` (`grid`, `free`) | `grid` |
| `--max-tokens N` | `max_tokens` (`free` only) | the grid plus slack |
| `--image FILE` | source images (server: image edits; C API: `images_json`) | none |
| `--nemo-image guidance=G` | `guidance` (the server's `guidance`): classifier-free guidance on the instruction | 1 (off); 2 with source images |
| `--nemo-image image_guidance=S` | `image_guidance`: guidance on the source images (a third sequence) | 1 (off) |
| `--nemo-image source_layout=L` | `source_layout`: `turn` (sources as earlier replies), `inline` (in the user turn) | `turn` |
| `--nemo-image source_prompt=TEXT` | `source_prompt`: the request the sources answer (`turn`) | empty |
| `--quant q8\|q4` | `quant` (the float32 release; GGUF files keep their own) | bf16 |
| `--image-tokenizer PATH\|REPO` | `tokenizer` | the checkpoint's, else NVIDIA's |
| `--nemo-image KEY=VALUE` | any of the keys above, and `vit_precision` (`bf16`, `f32`), `encoder` | |

In the library:

```rust
use oxinfer::nemotron_image::{GenerateOptions, LoadOptions, NemotronImage};
let m = NemotronImage::load("wiz0floyd/Nemotron-Labs-Image-30B-A3B-GGUF:Q4_0", &LoadOptions::default())?;
let o = GenerateOptions { width: 1184, height: 880, seed: 7, ..Default::default() };
let (imgs, stats) = m.generate("a red fox in the snow", &[], &o, &mut |p| {
    eprint!("\r{}/{}", p.step, p.steps);
    oxinfer::generate::Flow::Continue
})?;
let img = &imgs[0].image;
std::fs::write("fox.png", oxinfer::png::encode_rgb(img.width, img.height, &img.rgb))?;
```

`Generated` also holds the codes, the grid and the generated ids; `encode_image`, `decode_codes`,
`image_block`, `prompt_ids` and `parse_block` expose the stages, `image_tokenizer()` the ViT
(`vit::Vit::encode` / `decode`). The C API is `oxi_nemotron_image_load`, `_generate`, `_cancel`,
`_free` (`include/oxinfer.h`).

## Speed

On the M5 Max (18 CPU cores, 128 GB), the Q4_0 GGUF, other jobs running on the
CPU (`tools/bench/with-lock.sh` held for the GPU):

| | tokens | sampling | total |
|---|---|---|---|
| 1024x1024, Metal | 4,175 | 28.4-29.7 s (141-147 tok/s) | 30.2-31.6 s |
| 2 x 1024x1024, Metal (`--num-images 2`, both images in each GPU step) | 8,350 | 47.3 s (176 tok/s) | 50.4 s |
| 4 x 1024x1024, Metal (`--num-images 4`) | 16,700 | 70-76 s (219-238 tok/s) | 74-80 s |
| 1024x1024, CPU (`--device cpu`) | 4,175 | 102 s (41 tok/s) | 107 s |
| edit of a 1184x880 image (4,230 prompt tokens), guidance 2, Metal | 4,139 | 41.1 s (101 tok/s) | 46 s, prefill 3.7 s |
| 2 edits of it (`--num-images 2`, both images' 4 sequences in each GPU step) | 8,278 | 68.9 s (120 tok/s; one edit in the same session 44.9 s) | 74.5 s |
| 4 edits of it (8 sequences a step) | 16,556 | 134.9 s (123 tok/s) | 143.8 s |
| 1024x1024, the float32 release with `--quant q4`, CPU | 4,175 | 173 s (24 tok/s) | 3.1 min, after 3.9 min of loading |

Before the batched steps, the prefill plan and the experts fix below (the first Metal plan):
1024x1024 in 33.7 s (130 tok/s), 1184x880 in 32.0 s, 880x1184 in 30.0 s, 2 x 1024x1024 one after
the other in 59.7 s; the 1184x880 edit in 68 s (76 tok/s, prefill 12 s), a 1024x1024 edit in 78
to 82 s. A step of one image is 6.2 ms of GPU time out of 6.8 ms (`OXINFER_METAL_TRACE`): it reads
about 2.6 GB of weights (the visual rows of the head 0.56 GB of it), close to the memory's
bandwidth, so one image goes no faster; several images share the weight reads of every step but
their routed experts' (each token picks its own 6 of 128). A step of 4 images at 512x512 is
12.1 ms of GPU time (279 tok/s in all): about what reading the shared weights once (1.9 GB) and
each row's experts (0.78 GB a row) takes at the memory's bandwidth; per kind of op
(`OXINFER_PF_PROFILE=1`), the routed experts take a third of it, the Mamba-2 input projections a
sixth, the head a tenth.

The prompt takes about 1 s (0.1 s once the plans are built), the tokenizer's decoder 0.5 s on Metal (its matmuls and attention
on the GPU) and 4 to 7 s on the CPU. For comparison, the community GGUF's README reports 19 to 30
tok/s on a Radeon RX 7800 XT with Vulkan (experts on the CPU) and 100 to 120 tok/s with llama.cpp
on an A100.

**Decoding on the GPU.** Nemotron-H now has a whole-step Metal decode plan (`src/nemotron.rs`,
`gpu_graph`; `NemotronModel::forward` takes it for one-token steps, so the chat engine's
Nemotron-H models (Lightning, Transcribe) decode on it too, which was not measured here); it
needed operations of its own in the decode IR (`src/gpu_decode.rs`) and the Metal backend:

- `Op::Mamba2` (`dec_mamba2`): the selective state update, a threadgroup per head and a
  SIMD-group per row of its 64 x 128 state (each row read and written as whole cache lines), the
  `D` skip and the `silu(z)` gate; the grouped RMSNorm after it is `Op::Norm` with a weight row
  per group;
- `Op::GdnConv` with a bias (`dec_mamba_conv`), the causal convolution of `x | B | C`;
- `Op::Route` with sigmoid routing (`dec_topk`: `sigmoid(logit) + bias` chooses, the chosen
  sigmoids renormalized and scaled by 2.5);
- `Experts::relu2`, the ungated squared-ReLU experts (the pair kernels with the up matrix in
  the first slot, the second left unread in mode 9, and a squared-ReLU store), and
  `LinMode::Relu2` for the shared expert.

They run in every kind of plan: one-token steps, multi-token plans (`dec_mamba2_t`: a SIMD-group
per state row keeps its row in registers over the step's tokens; `dec_gdn_conv_t` with the bias;
`dec_rms_norm_rt` for the grouped norm, every token's rows in one dispatch; `dec_topk` and
`pf_topk` with the sigmoid routing; `mode 9` of the multi-token matvecs), prefill plans (the same,
`pf_conv` with the bias, the routed experts as grouped GEMMs with a squared-ReLU `silu_half`, the
shared expert a GEMM then `dec_ew_t` mode 6) and batched rows (`DecodePlan::step_rows`: each
row's recurrent ops on the one-token kernels with its own state). The squared-ReLU experts no
longer read their matrix twice (the pair kernels skip the second one in mode 9: the same bits,
half the work), which took Lightning's decoding from 93 to 153 tok/s
([Nemotron 3.5](nemotron.md#on-metal)). `NemotronModel::forward_rows_batch` runs one token of
each of up to 16 sessions; `--num-images N` (the server's `n`) runs the prompt once and the N
grids in lockstep on it, each image with its own seed: the same images as one at a time, byte
for byte (the batch's rows are the one-token steps' bits, `OXINFER_GPU_DECODE_CHECK=1` compares
each row with a one-token step of its session: 510 of 510 equal; image 0 of a 4-image batch is
the PNG of a single run with that seed). With guidance (edits, `guidance` / `image_guidance`)
the prompt and its guidance prompts run once, and every image's sequences (two or three each)
take each token together, 16 rows a step at most (8 guided images, 5 with image guidance): a
4-image edit gives the PNGs of 4 single edits with those seeds. The CUDA backend declines these ops and runs
Nemotron-H on the CPU. Inside the image grid only the visual tokens'
logits are computed: their 196,560 rows of the head (60% of the 0.94 GB Q8_0 head) are a
matrix of their own (`NemotronModel::load_head_rows`, borrowed from the GGUF file; on the CPU,
where the head is interleaved for its kernels, the whole head runs and is sliced).

## Agreement

`tools/ref_nemotron_image.py` runs NVIDIA's own `modeling_l24sq_vit.py` (from the checkpoint's
tokenizer folder, CPU PyTorch 2.15) and `examples/nemotron_image_check.rs` the same stages in
oxinfer:

| tokenizer, 320x192 image (12 x 20 codes) and 250x170 (padded to 256x176, 11 x 16) | codes equal | decoder output from the reference's codes |
|---|---|---|
| float32 weights (`vit_precision=f32`, `OXINFER_VIT_F32=1`) against float32 | 240 / 240 | max 1.6e-5, relative 9.5e-7 |
| bf16 weights (the default) against the reference with bf16-rounded weights | 238 / 240, 175 / 176 | relative 1.9e-3 / 1.6e-3 (64.5 / 69.3 dB) |

The bf16 difference is the CPU's bf16 matmuls, which round the activations too; the codes that
differ are near ties of the cosine.

The language model is `src/nemotron.rs`, checked against transformers in
[Nemotron](nemotron.md#agreement); the reference here runs it on vLLM with four H100s, which could
not be run, so the generation itself was compared by looking at the images (above) rather than by
logits. The GPU decode plan against the CPU path on the Q4_0 GGUF (`OXINFER_GPU_DECODE_CHECK=1`,
every step of a generation): mean KL per step about 5e-3 over the 327,680 logits, largest logit
difference about 0.5, the same top token at 27 of 30 steps; the CPU's matmuls round the
activations to int8 for the quantized blocks (the GPU's are float32), and the visual tokens'
distribution is flat (hundreds of near-equal candidates), which is where the top tokens differ.

`--unconstrained` reproduces the reference's protocol: asked for 1024x1024, the Q4_0 model wrote
the line break, `<|img_start|>height=1024 width=1024`, 64 rows of 64 and `<|img_end|>` (4,175
tokens), exactly the block the grid mode writes.
