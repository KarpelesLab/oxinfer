# Typed decisions (Jev models)

The Jev models (`autotrust/JEV-9B`, `autotrust/JEV-27B`, ...) answer a typed question about a
piece of text (or images: [Image decisions](#image-decisions)) with calibrated probabilities, in one
prefill pass and with no generation ("System 1"): `noul` (yes / no, probabilities for `["false", "true"]`), `score` (a 0-5 scale) or
`choice` (2-16 options labelled A-P, up to 256 with the extra single-token labels Q-Z, AA, AB,
...). The prompt is the `bare-v1` template:

```text
[kind] choice
[state] SKU AX-330 stock at 8% of safety level; supplier late twice this quarter.
[question] Supplier response for this scenario.
[options]
A) issue_warning
B) renegotiate
C) dual_source
D) maintain
[decision]:
```

The last token's final-norm hidden state goes through a small head (`head.safetensors`, 24 slots),
the active slots are divided by a per-kind temperature (`calibration.json`) and soft-maxed. The
backbone of JEV-9B / JEV-27B is the unmodified Qwen3.5-9B / Qwen3.8-27B; the decision skill is a
LoRA adapter (`adapter/`, r=16, about 160 / 435 MB) that oxinfer applies **unmerged, at run time,
on top of whatever base weights are loaded** (bf16, `--quant q8|q4`, pre-quantized, MLX or GGUF)
and only during a decision. One loaded model therefore serves chat (the base model, bit for bit:
the adapter's layers are skipped when it is off) and decisions. The bundles that ship merged
weights and read the decision from the LM head (`adapter_vllm/decision_head.json`, e.g.
`autotrust/JEV-35B`) work too.

```sh
# a Jev bundle (backbone, adapter and head; the repo id downloads all of it)
oxinfer --model autotrust/JEV-9B --decide noul \
  --state "Customer says the parcel arrived damaged and wants their money back." \
  --question "Is the customer asking for a refund?"
#   false  0.0092
# * true   0.9908  ########################################

# any matching base you already have, plus only the decision files of the bundle (~420 MB for
# JEV-27B: adapter, head and configs, not the 54 GB backbone)
oxinfer --model ~/.cache/oxinfer/hub/Qwen3.8-27B-oxinfer-q4 --decision-from autotrust/JEV-27B \
  --decide choice --state "SKU AX-330 stock at 8% of safety level; supplier late twice this quarter." \
  --question "Supplier response for this scenario." \
  --option issue_warning --option renegotiate --option dual_source --option maintain

# requests in the /v1/decide JSON format (an object, an array or one per line): one response per line
oxinfer --model autotrust/JEV-9B --decide-json requests.json
```

`--decision-from` (engine option `decision_from`) takes a directory or a repo id; the base must
match the bundle (hidden size, layer count, vocabulary, verbalizer tokens, and the shape of every
adapted layer), otherwise loading fails with a message saying which. As a library:

```rust
use oxinfer::decide::DecisionRequest;
use oxinfer::engine::{Engine, EngineOptions};
let mut opts = EngineOptions::default();
opts.set("quant", "q4")?;
opts.set("decision_from", "autotrust/JEV-9B")?;
let mut engine = Engine::load("Qwen/Qwen3.5-9B", &opts)?;
let d = engine.decide(&DecisionRequest::choice(
    "I was charged twice for my subscription this month.",
    "Which team should handle this ticket?",
    ["billing", "technical support", "sales", "account security"],
))?;
println!("{} ({:.3})", d.best, d.probabilities[d.best_index]);
```

A decision runs in a scratch context: the conversation of the engine is left as it was, so
`send` and `decide` can be interleaved (`examples/decide.rs` checks that the chat reply is the same
before and after a decision, and the same as without any adapter). `Engine::decide_json` and the C
API's `oxi_decide` take and return the JSON of the reference server's `POST /v1/decide` (`kind`,
`state`, `question`, `options` in; `options`, `probabilities`, `choice_index`, `choice`,
`adaptation` (`native` or `wide-labels`), `usage`, ... out); `oxi_decide_info_json` describes the
loaded assets.

Agreement with the reference (`transformers` + `peft`, unmerged adapter, the card's slot-head
readout; 10 questions: 3 noul, 2 score, 4- and 12-option choices and a 20-option wide-label choice
read through `adapter_vllm`'s LM-head LoRA), JEV-9B:

| oxinfer | vs reference f32 | | vs reference bf16 |
|---|---|---|---|
| | max \|Δp\| | max KL | max \|Δp\| |
| bf16 weights, CPU | 4.8e-4 | 5.3e-7 | 3.9e-3 |
| bf16 weights, Metal | 5.0e-4 | 5.8e-7 | |
| BF16 GGUF base (unsloth) + JEV-9B adapter, Metal | 4.6e-4 | 5.2e-7 | |
| `--quant q8` | 6.5e-3 | 1.2e-4 | |

(The bf16 reference itself differs from the f32 one by 4.0e-3.) Qwen3.5-9B with
`--decision-from autotrust/JEV-9B` gives the same probabilities as the JEV-9B folder, and JEV-27B on
the bf16 Qwen3.8-27B reproduces the model card (refund `true` 0.978, AX-330 `dual_source` 0.62);
on the Q4 conversion the largest difference to bf16 is 0.044 (KL 6.6e-3, from the 4-bit weights).
`autotrust/JEV-35B` (merged Qwen3.5-35B-A3B MoE, LM-head readout with 256 trained choice slots)
against `transformers` in bf16 (an f32 reference does not fit in 128 GB): median max |Δp| 4e-4,
worst 0.042 (KL 6.6e-3) on a near-tie score question; MoE routing flips near ties, so the CPU and
Metal runs of oxinfer differ from each other by up to 0.014 too. It decides in 0.25-0.45 s on
Metal for 45-160 tokens.

Latency (M5 Max, 18 threads; a decision is a prefill, so it scales with the prompt):

| model | device | 105-token prompt | 1461-token prompt | plain prefill of the same 1461 tokens |
|---|---|---|---|---|
| JEV-9B (bf16) | CPU | 1.31 s | 18.3 s | 17.8 s |
| JEV-9B (bf16) | Metal | 0.32 s | 2.63 s | 2.44 s |
| JEV-27B on Qwen3.8-27B Q4 (`decision_from`) | CPU | 5.2 s | 70 s | |
| JEV-27B on Qwen3.8-27B Q4 (`decision_from`) | Metal | 0.72 s | 6.2 s | 5.9 s |

The adapter costs 3-7 % of the prefill (low-rank products in f32 NEON on the CPU, after the base
matmul, also when the base matmul runs on the GPU). Not supported: Core ML models, and the
adaptive thinking of the reference server (`"thinking": "auto"`: System 2 reasoning folded into the
probabilities; the base model itself chats as usual in the same engine).

**LoRA on routed MoE experts.** An adapter may also target the routed experts of a MoE base
(Qwen3.5 / 3.6 MoE: `layers.N.mlp.experts`; Gemma 4 26B-A4B: `layers.N.experts`): each expert
gets its own low-rank update, applied to the tokens routed to it, on every CPU path of the
experts (decode, the batched kernels of short passes, grouped prefill) and with streamed experts.
Two adapter layouts are accepted, on either storage of the base experts (fused `gate_up_proj`, or
separate gate / up matrices as in GGUF):

- stacked parameters, what peft writes for the current transformers classes, whose experts are 3-D
  `[experts, out, in]` parameters: `LoraConfig(target_parameters=["experts.gate_up_proj",
  "experts.down_proj"])` saves nested wrappers `...experts.lora_A/B` and
  `...experts.base_layer.lora_A/B`, with `lora_A` `[experts * r, in]` and `lora_B`
  `[out, r * experts]`; the keys do not name the parameter, so it is told from the shape;
- one module per expert (`...experts.E.gate_proj`, `.up_proj`, `.down_proj`), ordinary pairs.

A prefill pass whose experts would run on the GPU runs them on the CPU while the adapter is on
(the device computes an expert in one submission, with no place for the side path; the dense
matmuls stay on the GPU, with theirs), and a model with an adapter attached decodes without the
GPU decode plan, as with any runtime LoRA. Not accepted, as an error at load (an expert module
that cannot be applied is never skipped): separate stacked `gate_proj` / `up_proj` parameters
(their wrappers have the same shape and cannot be told apart), a rank or expert count other than
the model's, both layouts on one weight, and DoRA / per-module ranks as for dense layers.

No published bundle adapts routed experts, so this was verified on synthetic adapters only:
`tools/moe_lora_tiny.py` makes a tiny random-weight Qwen3.5-MoE or Gemma 4 MoE checkpoint
(4 layers, 8 experts, top 3; hidden 128) and a random rank-4 peft adapter on the experts'
`gate_up_proj` / `down_proj` and on attention and MLP projections, and
`examples/moe_lora_check.rs` compares every position's logits of a 40-token prompt with peft's
unmerged forward (float32). bf16 weights, CPU, the adapter moving the logits by up to 9.6 (Qwen) /
4.4 (Gemma):

| | Qwen3.5-MoE: max \|Δlogit\| | max KL | Gemma 4 MoE: max \|Δlogit\| | max KL |
|---|---|---|---|---|
| one prefill pass | 7.1e-5 | 4.1e-10 | 6.0e-5 | 1.6e-10 |
| passes of 5 tokens, and token by token | 9.8e-5 | 3.6e-10 | 3.3e-5 | 5.5e-11 |
| adapter off, vs the base model | 1.6e-4 | 1.3e-9 | 4.3e-5 | 8.3e-11 |

The per-expert layout of the same adapter and the same checkpoint saved with one module per
expert give the same numbers in all four combinations. With larger random models (hidden and
expert width 1024, 4 experts, 160 tokens) on CUDA, where the base experts' prefill does run on the
GPU when the adapter is off: 5.1e-5 (Qwen) and 2.0e-4 (Gemma) with the adapter on, as on the CPU
(5.1e-5 / 2.0e-4). On Q8 / Q4 weights the batched expert kernels with the adapter match the
one-expert-at-a-time path bit for bit (logits of random weights quantized this coarsely are too
far from float32 to compare with the reference), and after a pass with the adapter the base
model's logits are bit for bit what they were. Not run: an adapter of this kind on real weights,
Metal, and streamed experts end to end (they take the same CPU paths).

## Gemma 4 26B-A4B bundles

`autotrust/GEV-26B-Decide` and `autotrust/JEV-Gemma4-26B-A4B` (the same weights published twice)
put System 1 on the unmodified `google/gemma-4-26B-A4B-it`: a LoRA (r=32) on the attention and
dense-MLP projections of all 30 layers (205 modules; the routed experts are not adapted), the
24-slot head and the temperatures. What differs from the Qwen bundles: the prompt starts with
`<bos>`, the head's logits are soft-capped like Gemma's own (`30 * tanh(z / 30)`, from
`judge_config.json`), and a choice with more than 16 options is a **tournament** by default, as in
the bundle's `serve_decide.py`: groups of at most 16 options are read natively, then a final pass
over each group's winner plus the best of the rest (16 in all), and the group shares are combined
as the reference does. `"strategy": "single"` (JSON) or `DecisionRequest::strategy` reads them in
one pass through the wide labels instead (GEV only: it needs `adapter_vllm/`'s LM-head readout;
JEV-Gemma4 has none). Images go where they appear, each as `<|image>` + its soft tokens +
`<image|>` with the processor's default budget, attending both ways within the image.

```sh
# only the decision files of the bundle (150 MB), on the Google base already on disk
oxinfer --model google/gemma-4-26B-A4B-it --mmap --decision-from autotrust/GEV-26B-Decide \
  --decide choice --state "Set an alarm for 6:30 tomorrow morning please." --question "Which intent?" \
  --option weather_query --option alarm_set --option alarm_remove --option play_music
oxinfer --model google/gemma-4-26B-A4B-it --mmap --decision-from autotrust/GEV-26B-Decide --decide-json requests.json
```

Agreement with transformers + peft (the card's readout, unmerged adapter, float32 compute on the
bf16 weights): the 10 text decisions of the JEV-9B check (3 noul, 2 score, 4- and 12-option
choices, a 20-option tournament) and 5 image decisions (fox photo and shapes, 1-2 images of 256-266
soft tokens, noul / score / 5- and 6-option choices):

| oxinfer | text: max \|Δp\| | max KL | images: max \|Δp\| | max KL |
|---|---|---|---|---|
| GEV-26B-Decide, bf16, CPU | 1.8e-6 | 7e-12 | 5.9e-3 (shapes; the others 5e-5) | 1.4e-4 |
| JEV-Gemma4-26B-A4B, bf16, Metal | 3.0e-3 | 2.6e-5 | | |
| 20 options with `"strategy": "single"` vs the reference through `adapter_vllm/` | 2.0e-3 | 2.5e-5 | | |

Prompt token counts match the processor's. The image difference comes from the vision tower's
input resize, as for chat (logits of image prompts differ by 0.03). On Metal (bf16, M5 Max) a text
decision takes 0.18-0.32 s for 46-139 prompt tokens, the 20-option tournament (3 passes, 364
tokens) 0.87 s, and an image decision 1.4 s (0.8 s of it in the vision tower; 2.5 s with two
images); on the CPU 0.4-0.9 s for text and 9-17 s with images.

## Image decisions

`autotrust/JEV-27B-VL` (Qwen3.8-27B with its vision tower, unchanged, plus JEV-27B's System 1 as
an unmerged `adapter_vllm/` LoRA with an `lm_head` readout) and JEV-9B's `vl/` bundle (the same
adapter as `adapter_vllm/` with the layer names of the multimodal class, for the unmodified
`Qwen/Qwen3.5-9B`) decide over images too. As in the reference server (`serve_decide.py`), the
state is a list mixing text and images, and each image goes where it appears:

```text
[kind] noul
[state] Photo: <|vision_start|><|image_pad|> x 256<|vision_end|>
[question] Is this scenario one where: the image shows a wild animal?
[options]
false
true
[decision]:
```

No chat template, no system prompt: text parts as they are, an image as its vision tokens (M-RoPE
positions as in chat), then the readout, adapter, bias and temperatures of the text decisions.
The head never saw an image in training, so these decisions are zero-shot. The reference server
takes images only (no video); so does oxinfer.

```sh
# JEV-9B on the multimodal Qwen3.5-9B: only vl/ is fetched from autotrust/JEV-9B (org/name/subdir)
oxinfer --model Qwen/Qwen3.5-9B --decision-from autotrust/JEV-9B/vl --decide choice \
  --image scene.png --state "Task: pick up the red cube." \
  --question "Where is the red object relative to the blue one?" --option left --option right

# JEV-27B-VL brings its own assets; JSON requests place images anywhere in the state
oxinfer --model autotrust/JEV-27B-VL --decide-json requests.json
```

`--image` (repeatable) puts images before the state text; `--max-soft-tokens N` caps the tokens of
an image (default: the processor's limits; each token covers 32x32 pixels). In JSON (`--decide-json`,
`Engine::decide_json`, `oxi_decide`) an image part is `{"image": REF}` or `{"type": "image_url",
"image_url": {"url": REF}}`, with REF a base64 `data:` URL, an `http(s)://` URL (fetched with the
`hf` feature's client), a `file://` URL or a path; `"attachments": {"images": [...],
"max_soft_tokens": N}` is the JSON form of `DecisionRequest::attachments`. In Rust:

```rust
let d = engine.decide(
    &DecisionRequest::noul(
        serde_json::json!(["Photo: ", {"image": "photo.jpg"}]),
        "Is this scenario one where: the image shows food or cooking?",
    ),
)?;
// or: DecisionRequest::noul("caption text", "...").with_images(["photo.jpg"])
// ImageSource::Bytes(png).to_part() makes a state part from encoded bytes
```

A text-only checkpoint (the `autotrust/JEV-9B` folder itself has no vision tower) refuses images
with a message naming the multimodal base to load instead.

Agreement with the reference server's math on `transformers` + `peft` (HF processor, unmerged
`adapter_vllm`, LM-head logprobs over the verbalizer tokens + bias, per-kind temperature; f32
compute on the bf16 weights), 11 image decisions (2-3 noul, 2 score, 2- and 4-option, two
12-option and a 20-option wide-label choice; one or two images of 140-396 vision tokens, as data
URLs, `image_url` parts and paths) and the 5 text decisions of the text check:

| oxinfer (bf16 weights) | images: max \|Δp\| | max KL | text: max \|Δp\| |
|---|---|---|---|
| Qwen3.5-9B + JEV-9B `vl/`, CPU | 3.7e-6 | 1.6e-8 | |
| Qwen3.5-9B + JEV-9B `vl/`, Metal | 1.1e-3 | 5.7e-6 | 1.3e-4 |
| JEV-27B-VL, CPU | 8.3e-6 | 4.5e-10 | |
| JEV-27B-VL, Metal | 6.0e-4 | 9.4e-7 | 8.7e-5 |

Prompt token counts match the processor's exactly; the Metal differences come from its half-precision
matrix units (`gpu_exact` removes them). Text decisions and chat give the same output as before
images were supported, bit for bit.

Latency of one image decision (512x512 photo = 256 vision tokens, 298 prompt tokens; M5 Max, while
other jobs loaded the machine, so upper bounds): Qwen3.5-9B + `vl/` 1.0-1.5 s on Metal (vision tower
0.4-0.6 s, prefill 0.6-0.9 s) and 7.4 s on the CPU; JEV-27B-VL (bf16) 3.3 s on Metal (tower 0.6 s)
and 22 s on the CPU.
