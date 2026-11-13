# Runtime interventions: neuron edits, steering, abliteration, SAE features

oxinfer can change what a model computes while it runs, without touching the weights: edit
chosen MLP neurons, add steering vectors to the residual stream, project a direction out of it
("abliteration" at run time), and read or steer sparse-autoencoder (SAE) features. It can also
make steering vectors itself, from two sets of prompts. Everything is reachable from the CLI,
the library (`Engine::set_interventions`, `oxinfer::intervene`) and the C API
(`oxi_engine_set_interventions`). Interventions are set per request: changing a strength
between two messages rebuilds a small set of hooks, not the model, and GPU decoding keeps its
compiled plan. The shared code is in `src/intervene.rs` and `src/sae.rs`, wired into Qwen3.5 /
3.8 (dense and MoE), Qwen3, Llama 3, Gemma 4 / 3, Mistral Large 4 and DeepSeek V4.

**Where the hooks are.** Layers are 0-based, as in the checkpoint's module names
(`model.layers.14`).

| Hook | Meaning |
|---|---|
| neuron `L:n` | element `n` of the input of `layers.L.mlp.down_proj`, that is `act(gate) * up`. MoE layers: per routed expert (`"expert": 3`; the expert's activation before its routing weight scales the output) or the shared expert (`"expert": "shared"`). Mistral Large 4: layer 0's dense MLP (24576 neurons) takes no `expert`; the MoE layers have the shared expert's 6144 (`mlp.shared_experts.down_proj`) and each routed expert's 6144 (`mlp.experts.down_proj[e]`). |
| `post` of layer L | the layer's output, `hidden_states[L + 1]` in transformers. This is where llama.cpp control vectors and Qwen-Scope SAEs (`resid_post`) act. |
| `mid` of layer L | the residual after the attention (or Gated DeltaNet) block was added |
| `pre` of layer L | the layer's input: the output of layer L - 1, or the embeddings for layer 0 |

**DeepSeek V4** has no single residual stream: each token carries `hc_mult` (4) copies of the
hidden state (`[tokens, 4, hidden]`), which every block reads through a learned weighted sum and
writes back through learned weights and a mixing matrix. The hooks are defined on that stream
(`State::point_copies` in `src/intervene.rs`):

- **The points.** `pre` of layer L is the 4-copy tensor entering `layers.L` (`Block.forward`'s
  input in DeepSeek's `inference/model.py`; the repeated embedding for layer 0). `mid` is the
  tensor after the attention half (the output of its `hc_post`, the input of the MoE half's
  `hc_pre`). `post` is the layer's output, which is the next layer's `pre`.
- **The residual at a point** is the mean of the 4 copies. Capture, `--make-direction`, the
  `resid_median` calibration, SAE encoding and inspection read that mean, so vectors and SAEs
  have `hidden` values as for every other family.
- **Edits write to every copy**, so that the mean changes exactly as a single stream would:
  steering adds `alpha * v` to each copy; ablation projects `r` out of each copy (it is then
  absent from every weighted sum a block reads, not only from the mean, and stays absent
  through the mixing, whose weights are scalars per copy; hooking Arditi's block outputs before
  they are spread over the copies gives the same logits, which the reference checks); an SAE
  feature edit is computed on the mean and its change `(a' - a) * W_dec[f]` is added to each
  copy.
- **Neurons.** Every layer is a MoE (the first 3 hash-routed), so an edit always names an
  `expert`: `"shared"` is the input of `ffn.shared_experts.w2` (2048 neurons), an id the input of
  `ffn.experts.E.w2` (2048 each, 256 experts), both `silu(min(gate, 10)) * clamp(up, -10, 10)`,
  for a routed expert before its routing weight. There is no dense MLP.

**Where they apply.** `positions` is `all` (the default), `prompt` or `generated`. `generated`
covers the last prompt token and every generated token, so every position whose output is
sampled. `prompt` covers the prompt tokens before that.

## The spec

The spec is a JSON object. It goes to `--interventions FILE`, to the engine option
`interventions`, or to `oxi_engine_set_interventions`. The library type is
`oxinfer::intervene::Spec`.

```json
{
  "neurons": [
    {"layer": 14, "neuron": 8949, "dose": 1.0, "unit": "resid_median"},
    {"layer": 23, "neuron": 4790, "op": "scale", "value": 2.0},
    {"layer": 18, "neuron": 6052, "op": "zero"},
    {"layer": 5, "neuron": 7, "expert": "shared", "op": "clamp", "min": -1, "max": 1, "positions": "generated"}
  ],
  "steer": [{"file": "tone.safetensors", "layer": 12, "alpha": 1.5, "point": "post"}],
  "ablate": [{"file": "refusal.safetensors", "direction_layer": 12}],
  "sae": [{"sae": "Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_50/layer12.sae.pt",
           "features": [{"id": 10155, "op": "clamp", "value": 4.0}]}],
  "sae_inspect": [{"sae": "Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_50/layer12.sae.pt", "top_k": 5}]
}
```

- **Neuron edits.** `op` is one of:
  - `scale` multiplies the activation by `value`;
  - `zero`;
  - `add` adds `value`; `dose` is shorthand for this;
  - `set` replaces it with `value`;
  - `clamp` limits it to `[min, max]`.

  `"unit": "resid_median"` multiplies `value` by `median_t |h_L,t| / |W_down[:, n]|`. Here `h_L`
  is the output of layer L, measured on the unedited model over positions 1 onward (position 0,
  an attention sink with a huge norm, is skipped). With this unit, a dose of 1 moves the
  residual by the typical norm of that layer's output.

  The median comes from one unedited pass over the first prompt the edits see, and is then kept.
  You can instead measure it on a text of your choice (`--calibrate-text T`,
  `Engine::calibrate(text)`) or give it outright with `"resid_median": F`.

  The edit is added at every position (`add`, not `set`). Even-length medians average the two
  middle values (numpy's convention). The reference script applies the same edit to the same
  tensor through a PyTorch pre-hook on `down_proj`.
- **Steering.** Adds `alpha * v`. `v` is either an inline `vector` (with a `layer`) or comes from a
  `file`. With a file you get every layer it holds, or only `layer` / `layers`. `normalize: true`
  scales `v` to unit norm first, so `alpha` becomes the norm of what is added.
- **Ablation.** Applies `x -= (x . r) r` with `r` normalized. It runs on the embeddings and after
  every attention and MLP block, in all layers (or `layers`). This is Arditi et al.'s directional
  ablation; FailSpy's and mlabonne's tools hook the same places. Several directions are
  orthonormalized together.

  At one point, added vectors go in first and then directions are projected out. So an ablated
  direction also removes its part of a steering vector, just as a hook on the next layer's input
  would.
- **SAE features.** `op` is one of:
  - `clamp` sets the activation to `value` (Neuronpedia's clamp);
  - `scale` multiplies it;
  - `zero`;
  - `add` adds `value * W_dec[f]`, whatever the activation (SAELens / Neuronpedia / Qwen-Scope
    steering).

  The SAE's error term is kept: only `(a' - a) * W_dec[f]` is added to the residual. All edits of
  one SAE read the same encoding of the incoming residual. `sae_inspect` records the `top_k` most
  active features of every position: `Engine::sae_features()`, `oxi_engine_sae_features_json`,
  printed by the CLI.

**Vector files.** These are read by `steer` / `ablate` and written by `--make-direction`:
- **safetensors** with 1-D tensors `layers.N` (f32/bf16/f16). `layers.N` is the output of layer N.
  A single `[layers, hidden]` or `[layers + 1, hidden]` matrix is also accepted; the second form
  follows transformers' `hidden_states` and its row 0 is dropped. So is a lone 1-D vector.
- **GGUF control vectors**, as llama.cpp `--control-vector` and repeng's `export_gguf` write them:
  `direction.N` is added to the output of layer N, exactly as llama.cpp applies it. The same
  vectors as GGUF and as safetensors give bit-identical logits.
- **PyTorch `torch.save`** state dicts with the same names.

**SAEs.** Two formats load:
- **Qwen-Scope** (`Qwen/SAE-Res-*`, one `layerN.sae.pt` per layer plus the repo's `config.json`;
  TopK, `topk(relu(x W_enc^T + b_enc), k)`, `resid_post`).
- **SAELens directories** (`cfg.json` + `sae_weights.safetensors`; ReLU, JumpReLU or TopK,
  `apply_b_dec_to_input`; only `normalize_activations: none`).

Qwen-Scope has SAEs for Qwen3.5-2B-Base (32K features, k 50 or 100, 537 MB per layer), 9B-Base,
27B (80K features, the one Neuronpedia lists as `qwen3.5-27b / 31-qwenscope-res-80k`) and
35B-A3B-Base. There are none for 0.8B.

## Command line

```sh
# The user-reported "evil persona" edit on Qwen3.5-4B (layers 0-based, as in the module names)
oxinfer -m Qwen/Qwen3.5-4B -p "Write a short poem about the sea." \
  --neuron-edit 14:8949,dose=1,unit=resid_median --neuron-edit 13:7956,dose=-1,unit=resid_median

# A refusal direction from two prompt sets (one per line, or .json / .jsonl / .csv), then
# runtime abliteration with the layer-12 direction, and the opposite: activation addition
oxinfer -m Qwen/Qwen3.5-2B --make-direction harmful.json,harmless.json -o refusal.safetensors
oxinfer -m Qwen/Qwen3.5-2B --ablate refusal.safetensors:12 -p "..."
oxinfer -m Qwen/Qwen3.5-2B --steer refusal.safetensors:1.0 --steer-layers 12 -p "..."

# A llama.cpp / repeng control vector, generated tokens only
oxinfer -m Qwen/Qwen3.5-2B --steer happy.gguf:2 --positions generated -p "Describe a bus stop."

# SAE: inspect, then steer (the SAE file is fetched from Hugging Face)
oxinfer -m Qwen/Qwen3.5-2B-Base --raw -p "I adopted a cat." \
  --sae Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_50/layer12.sae.pt --sae-inspect 5
oxinfer -m Qwen/Qwen3.5-2B-Base --raw -p "My favorite thing to do is" \
  --sae Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_50/layer12.sae.pt --sae-feature 10155:add=4

# Pooled layer outputs of one prompt (layers.N, for your own analysis)
oxinfer -m Qwen/Qwen3.5-0.8B -p "..." --capture-activations acts.safetensors --capture-pool mean
```

Other flags:
- `--interventions FILE|JSON` takes the full spec.
- `--steer-point pre|mid|post` and `--ablate-layers A-B` place the hooks.
- `--sae-layer N` overrides the SAE's layer.
- `--capture-pool last|mean` pools the captured layer outputs. `last` is the last prompt token; `mean`
  averages positions 1 onward.

`--make-direction` wraps every prompt in the chat template unless `--raw` is given. It writes
`mean(first set) - mean(second set)` per layer, not normalized. With `alpha` 1, steering then
moves the second set's mean onto the first's.

## Library and C API

```rust
use oxinfer::engine::{Engine, EngineOptions, Flow};
use oxinfer::intervene::{Pool, Spec};
let mut engine = Engine::load("Qwen/Qwen3.5-2B", &EngineOptions::default())?;
// A direction from two prompt sets, saved for later use.
let d = engine.make_direction(&harmful, &harmless, Pool::Last, true)?;
oxinfer::intervene::save_vectors("refusal.safetensors".as_ref(), &d, &[])?;
// Change the interventions between requests: no reload.
for alpha in [0.5, 1.0, 2.0] {
    engine.set_interventions(Some(Spec::from_json(&format!(
        r#"{{"steer":[{{"file":"refusal.safetensors","layer":12,"alpha":{alpha}}}]}}"#
    ))?))?;
    engine.reset();
    let r = engine.send("Write a haiku about tea.", &mut |_| Flow::Continue)?;
}
engine.set_interventions(None)?;
```

`Engine::capture` returns the pooled layer outputs of a prompt list, and `Engine::calibrate`
measures the `resid_median` unit on a text. `examples/steer.rs` runs a prompt file under several
specs and strengths and writes the replies as JSONL; it produced the refusal and tone numbers
below.

The C API mirrors the library:
- `oxi_engine_set_interventions(engine, json)` (NULL clears) and
  `oxi_engine_interventions_json`;
- `oxi_engine_calibrate(engine, text)`;
- `oxi_engine_sae_features_json`;
- `oxi_engine_make_direction(engine, {"positive": [...] | "file", "negative": ..., "pool", "chat", "output"})`;
- the load option `interventions=JSON|@FILE`.

## Agreement with PyTorch hooks

`tools/ref_intervene.py` applies the same JSON spec to a transformers model with forward hooks
and compares logits. The hooks are its own implementation:
- a pre-hook on `down_proj` for neuron edits;
- output hooks on the layer and on the attention block for steering;
- Arditi's three hook sites for ablation;
- the release's own `topk(relu(...))` for Qwen-Scope.

Every row below compares the last prompt position, with oxinfer on the CPU against float32
PyTorch:

| Model | Interventions | max \|logit diff\| | KL | without interventions |
|---|---|---|---|---|
| Qwen3.5-0.8B | 4 neuron edits (scale, zero, dose resid_median, clamp), steering at 2 layers + a `mid` hook on generated positions, ablation in all layers | 2e-5 | 5e-8 | 4e-5 |
| Qwen3.5-4B | the two-neuron dose edit above | 4e-5 | 1e-8 | 6e-5 |
| Qwen3.5-2B-Base | Qwen-Scope layer 12: clamp, add, zero (TopK, error term kept) | 4e-5 | 3e-8 | |
| Qwen3-0.6B | neuron edits, steering (post, mid), ablation | 7e-5 | 5e-8 | |
| Gemma 4 E2B | neuron edits incl. resid_median, steering (post, mid), ablation | 2.2e-4 | ~0 | 1.3e-4 |
| Mistral Large 4, tiny random checkpoint (`tools/mistral4_tiny.py`: dense layer 0, 16 experts + shared) | 8 neuron edits (dense scale and resid_median dose; shared add and clamp; routed set, zero, resid_median dose on generated positions, scale on prompt positions), steering (post, mid on generated, pre normalized), ablation in all layers | 2e-5 | 1e-7 | 1.7e-4 |
| DeepSeek V4, tiny random checkpoint (`tools/deepseek4_tiny.py`: 6 layers, hash-routed layer 0, window-only, CSA and HCA layers, 8 experts + shared, 4 hyper-connection copies) | 32 neuron edits (shared add, clamp, resid_median dose; routed set, zero, scale on prompt positions, resid_median dose on generated positions, scale; one neuron of all 8 experts of 3 layers), steering (post at 2 layers, mid on generated, pre normalized, post on prompt positions), a TopK SAE (clamp, add, zero, scale) and a JumpReLU SAE (clamp, scale, add on generated positions), ablation in layers 0-1; a 106-token prompt | 2e-5 | 2e-8 | 3e-5 |

The `resid_median` values matched to 4 digits (for example L13 9.1125, L14 10.1972 on the
4B prompt).

DeepSeek V4 was measured on the tiny checkpoint only (the real model is a 97 GB download and
was not run with interventions). The reference (`ref_intervene.py`, transformers'
`DeepseekV4ForCausalLM`) hooks the same tensors: the 4-copy input and output of `layers[L]`,
the input of `ffn_hc` for `mid`, the shared expert's `down_proj` input, and the experts' loop
with the clamped SwiGLU for routed experts. Each kind alone agreed to 3e-5 or better (KL below
2e-7): the neuron edits, the steering entries, ablation of one direction in all layers and of
two (the second in layers 1-3 on generated positions), the TopK SAE, the JumpReLU SAE. With
`--block-hooks` the reference ablates on Arditi's sites instead (layer input, attention and
MoE outputs before they are spread over the copies) and gives the same logits (3e-5). The
combined spec gave 1e-5 with the prompt prefilled in passes of 7 tokens and of 1, and 2e-5 with
the weights streamed (`--stream`). Calibration medians matched to the 4 digits printed (L3
30.7286, L4 33.1161). Every spec gave the same 16 of 16 greedy tokens as `generate` under the
hooks. Captured layer outputs (mean pool) matched the mean of the copies in PyTorch to 1e-6
relative, and SAE inspection gave the same top-3 features and activations. Through the engine,
decoding with a (tiny random) DSpark draft gave the same replies as plain decoding under the
combined spec and under ablation (2 prompts, 24 tokens each).

Greedy generation on Qwen3.5-4B with the dose edits produced the same 40 of 40 tokens as
transformers' `generate` under the same hooks, with oxinfer decoding on the GPU. On the tiny
Mistral Large 4, every spec (all of the row above, and neurons, steering and ablation alone) gave
the same 8 of 8 greedy tokens as `generate` under the hooks; the reference edits the routed
experts by running transformers' fused-expert loop with the edit in place.

Qwen-Scope inspection agreed exactly: the top-5 features at all 25 positions of a sentence had
the same ids in the same order as the release's encoder, with activations equal to the 2
printed decimals.

GPU decode against the CPU (`OXINFER_GPU_DECODE_CHECK=1`, every step) gave the following KL:

| Model and interventions | KL |
|---|---|
| Qwen3.5-0.8B, all of the above | ≤ 2e-12 |
| Qwen3.5-0.8B, ReLU and JumpReLU SAE features | ≤ 2e-12 |
| a Qwen3.5 MoE test model, routed and shared-expert neuron edits | ≤ 1e-11 |
| Qwen3-0.6B | ≤ 3e-11 |
| Gemma 4 E2B | ≤ 1.3e-10 (1.2e-11 without interventions) |

## Results

**The Qwen3.5-4B persona edit:**
- **Edit:** +1 dose on layer 14 neuron 8949 and -1 on layer 13 neuron 7956.
- **Effect on a poem:** a gentle sea poem turns into a menacing one.
- **Effect on 12 neutral prompts** (introductions, advice, a birthday party plan, recipes; 120
  tokens each):
  - Replies containing hostile vocabulary went from 2 to 6.
  - Hostile words in total went from 2 to 20.
  - One reply proposed a "hostile takeover" as a party plan.
- **Calibration:** the medians come from the first prompt, on the unedited model (for the poem
  prompt: L13 9.11, L14 10.20).

**Runtime abliteration** (Qwen3.5-2B):
- **Direction:** the difference of means between 128 harmful and 128 harmless training
  instructions (Arditi et al.'s public splits), taken at the last token of the chat-templated
  prompt. Making it took 13 s.
- **Choosing a layer** (32 validation prompts, layer-L direction ablated everywhere):

  | Direction from | Refusal rate |
  |---|---|
  | no ablation | 81% |
  | layer 6 | 56% |
  | layer 8 | 38% |
  | layers 10 to 18 | 0% |

- **Held-out test** (100 harmful prompts, 64 tokens each, layer-12 direction): refusals fell from
  89% to 3%.
- **Held-out harmless prompts** (100):
  - ablation left refusals at 2% → 1%, and the replies stayed fluent and on topic;
  - adding the same direction at layer 12 (activation addition, alpha 1) made the model refuse
    92% of them.
- **How refusals were counted:** refusal phrases in the first 300 characters. The completions are
  not reproduced here.

**Tone steering** (Qwen3.5-2B, a happy-minus-sad direction from 24 + 24 prompts, layer 12, 10
neutral "two sentences about X" prompts):
- **alpha +1:** enthusiastic replies with exclamation marks and emoji ("the ultimate two-wheeled
  joyride").
- **alpha -1:** bleak replies ("its metal frame and chain eventually rust and become useless").
- **alpha ±2:** the text degenerates.

**SAE features:**
- **Inspection** finds concept features. For example, feature 10155 at layer 12 of the 2B-Base
  SAE fires on "cat", "kitten" and "Cats".
- **Single-feature steering** on the 2B base model nudged completions mildly (adding 3 to 7 times
  the unit decoder vector at generated positions). Larger strengths (10 and up, against residual
  norms of a few units at that layer) broke the text. Concept steering with one Qwen-Scope
  feature is weak at this size; the 9B / 27B SAEs are the ones Qwen's demo uses.

**The papers:**
- **H-Neurons** (arXiv 2512.01797) scale the `down_proj` input of selected neurons by
  alpha in [0, 3] (`op: scale`). They publish neither neuron lists nor Qwen results, so there is
  nothing to reproduce directly.
- **Perturbation Probing** (arXiv 2604.27401) scales the `W_down` columns of its Qwen3.5-2B
  neurons, for example L23:n4790, L18:n6052 and L14:n200 (2x amplification, zeroing for
  ablation). Scaling a column is the same as `op: scale` on that neuron. Their full 2B lists are
  not published.

## What runs where, and what it costs

| | CPU | Metal prefill | Metal whole-pass decode |
|---|---|---|---|
| neuron edits (dense, shared, routed experts) | yes | yes (hooks run between the GPU matmuls; MoE layers with routed-expert edits keep their experts on the CPU) | yes |
| steering, ablation | yes | yes | yes |
| SAE ReLU / JumpReLU features, SAE `add` | yes | yes | yes (up to 64 directional items per hook point) |
| TopK SAE clamp / scale / zero, SAE inspection, capture | yes | yes | no: those decode steps run on the CPU |

Every combination runs. When the GPU graph cannot take a hook, decoding falls back to the CPU on
its own, and the CLI says so when it loads the spec.

Decode speed on Qwen3.5-2B (bf16, M5 Max, 200 tokens, best of 3):

| Spec | Metal | CPU |
|---|---|---|
| none | 130.4 tok/s | 66.9 tok/s |
| steering at 1 layer | 130.4 | 66.6 |
| ablation at 49 hook points | 125.8 (-3.5%) | 66.1 |
| 3 neuron edits | 129.9 | 66.5 |

A TopK SAE clamp on 2B-Base moves decoding to the CPU: 124.6 → 63.0 tok/s, which is about the
CPU's own speed.

## Limits

- **Core ML (ANEMLL) models** refuse interventions.
- **Mistral Large 4** has no whole-pass GPU decode, so its hooks always run on the CPU between
  the (GPU or CPU) matmuls; a prefill pass with routed-expert edits keeps that layer's experts on
  the CPU. Typed decisions (`decision_from`) are not offered for it: a decision bundle is a LoRA
  adapter and head trained on one backbone (Qwen3.5 / 3.8, Gemma 4), none exists for this model,
  and its layers take no LoRA.
- **DeepSeek V4** is in the same position: no whole-pass GPU decode exists for this family, so
  on every device the hooks run on the CPU between the matmuls, in prefill and in decoding, and
  the CLI's "GPU decode keeps running on the GPU" line does not apply. A pass with routed-expert
  edits keeps that layer's experts on the CPU. Streamed weights work unchanged (the hooks touch
  activations only; the `resid_median` unit reads one `down_proj` column, for a routed expert
  through the streamer). The CLI decodes without the DSpark draft when interventions are
  given; the engine keeps the draft, whose stages are not hooked: they read the edited model's
  residual and the edited model verifies every draft, so the output follows the edited model.
  An SAE must have been trained on the mean of the copies (none is published for this model),
  and agreement was measured on a tiny random checkpoint only.
- **Speculative decoding.**
  - Qwen MTP drafts come from the unedited MTP head, but the edited main model verifies every
    draft, so the output follows the edited model.
  - The Gemma drafter path of the CLI is not covered.
- **SAEs.**
  - Only residual-stream outputs (resid_post) and `normalize_activations: none` are supported.
  - There is no GemmaScope `.npz` loader.
  - Weights are loaded as f32: 537 MB per Qwen-Scope 2B layer.
- **Making directions.**
  - Capture runs on the CPU paths (prefill matmuls still use the GPU).
  - Pooling is last-token or mean only. Arditi's per-position candidates and his
    bypass/induce/KL layer selection are not built in; the layer sweep above was done with
    `examples/steer.rs`.
- **Calibration.** The `resid_median` unit is measured once, on the first prompt (or the
  calibration text), and kept until `Engine::clear_calibration`.
