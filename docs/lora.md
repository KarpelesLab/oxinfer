# LoRA adapters for language models

oxinfer applies LoRA adapters to the language models at run time without merging them: the base
weights stay as they were loaded (bf16, Q8 / Q4, MLX, any GGUF block format), and each adapted
layer adds `strength * scale * B (A x)` to its output. The low-rank products run in f32 after the
base matmul: on the CPU, or in the GPU decode and prefill plans (Metal, CUDA) as side paths of the
matvecs (see [GPU plans](#gpu-plans)). On the CPU an adapter at strength 0 runs exactly the base
model's code path. Two formats are read, for the same families:

- **PEFT**: a folder with `adapter_config.json` and `adapter_model.safetensors`, as `peft` saves it,
  or a Hugging Face repo id holding one;
- **llama.cpp GGUF adapters**: `general.type = adapter` and `adapter.type = lora` files, as
  `convert_lora_to_gguf.py` writes them and `llama-cli --lora` / `--lora-scaled` reads them.

```sh
# A GGUF adapter on a GGUF base of the same architecture (strength 1, as llama.cpp's --lora)
oxinfer --model AtomicChat/Inkling-2-Mini-GGUF:AD-IQ2_M \
  --lora AtomicChat/Inkling-2-Mini-Uncensored-GGUF/Inkling-2-Mini-Uncensored-LoRA.gguf -p "Hello"
# A PEFT folder at strength 0.5 (llama.cpp's --lora-scaled FILE 0.5), and a second adapter
oxinfer --model Qwen/Qwen3-8B --lora ./my-adapter:0.5 --lora org/other-adapter
```

`--lora FILE[:SCALE]` is repeatable, and each adapter's updates are added in order. It is the same
flag as the image models' LoRA option: for a diffusion model it still names a kohya or diffusers
LoRA, which is merged as before. `--lora-merge` merges language-model LoRAs into the weights at
load; for MOSS-Image it merges low-rank LoRAs, as before. `FILE` may be a local file or folder,
`org/name/path/file.gguf` on Hugging Face, or `org/name` (or `org/name/subfolder`) holding a PEFT
adapter.

## Interfaces

| Where | How |
|---|---|
| CLI | `--lora FILE[:SCALE]` (repeatable), `--lora-merge`; also with `--prompt-cache` / sessions and `--decide` |
| Library | `EngineOptions::loras` (`Vec<oxinfer::lora::LoraSpec>`) and `lora_merge`, or `opts.set("lora", "FILE[:SCALE]")` once per adapter (`none` clears the list, a JSON array of specs adds them all) and `opts.set("lora_merge", "1")`; `Engine::loras()` lists the attached adapters (`oxinfer::lora::Attached`: path, format, rank, alpha, updates, merged count, bytes), `Engine::set_lora_strength(i, s)` changes a strength (0: off) and starts the conversation again |
| Lower level | `oxinfer::lora::attach_specs(&mut model, &specs, model_path, merge)` on a `QwenModel`, Gemma `Model` or `Inkling2Model` (the `Adaptable` trait); `LoraAdapter::open` / `load` / `load_gguf` |
| C API | `lora=FILE[:SCALE]` lines (repeatable) and `lora_merge=1` in the options of `oxi_engine_load`; `oxi_engine_loras_json`, `oxi_engine_set_lora_strength(engine, index, strength)` |
| Server, per request | `MODEL@PROFILE` runs with the profile's adapter ([learning from outcomes](learning.md)) |
| Training | `oxinfer lora-train --model M --profile DIR` writes a PEFT folder trained from a profile's episodes and outcomes (dense Qwen3 and Llama 3, CPU; [training a LoRA from episodes](learning.md#training-a-lora-from-episodes)) |
| Server | the same model options: `qwen=Qwen/Qwen3-8B,lora=./adapter:0.5,lora=./other`; in a JSON config, `"options": {"lora": ["./a", "./b:0.5"]}` (an array) or one string |

The prompt cache keys snapshots by the adapters and their strengths, so a cached conversation of
the base model is never restored into an adapted one (or the reverse).

## Adapters on a loaded model

`EnginePool::attach_adapter(name, path, strength, version)` puts an adapter on a model that is
already loaded, without reloading its weights, and `detach_adapter(name)` takes it off. Such an
adapter sits behind a switch of its own, off by default: `EnginePool::acquire_with(messages,
Some(name), ..)` turns it on for one request. The server uses this for the profiles of
[learning from outcomes](learning.md) (`MODEL@PROFILE`): one adapter per profile, on the shared
weights. Adapters given at load (`lora=`) are not affected: they stay on for every request.

What it does, and what that costs:

- attaching and detaching wait for the requests in flight on the model to end (none sees the
  change), and requests arriving meanwhile wait;
- every session of the pool then starts again: conversations in memory are read again on their
  next request (or restored from the prompt cache, which keys its snapshots by the adapter's
  path, version and strength as well);
- the GPU decode and prefill plans are dropped and compiled again on their next use, since
  they hold the adapters' side paths (see below);
- the switch is the model's, so the requests running at one time all use the same adapter, or
  none: a request for another one waits for them to end (docs/learning.md, "Requests at the same
  time");
- a detached or replaced adapter is retired, not freed: its updates add nothing on the CPU and
  are left out of the plans compiled from then on, but its factors stay in memory until the model
  is unloaded. An adapter attached and switched off is still in the plans, where its side paths
  add zeros.

Attaching is for Qwen-path and Gemma models decoded without MTP or a DSpark draft, not streamed
from disk, without interventions or decision assets; the others return an error and stay as they
were. Updates are runtime ones (no merge). An adapter that fails part of the way (a module of
another shape) is retired as a whole. MTP-layer modules of such an adapter are not applied
(these models decode without the MTP layer).

Checked (`engine::pool_slots::tests`, tiny Qwen3.5, CPU): requests with the adapter give the
tokens of a model loaded with it, requests without it the base model's, with two adapters
attached, after a replacement, a failed load and a detach; requests with different adapters do
not overlap. On the CUDA backend, see docs/learning.md, "What was checked". Gemma and Metal were
not run.

## GPU plans

Qwen-path and Gemma models with runtime adapters keep their GPU decode plans (single-token,
multi-token for MTP / DSpark / lookup verification, Metal and CUDA) and their whole-pass Metal
prefill plans. Each adapted linear op gets two ops in the plan graph (`gpu::decode::Op::LoraDown`
/ `LoraUp`): `u = A x` as partial sums over chunks of 1024 inputs, which only reads the input and
runs beside the matvec, then `y += sum_k u_k S_k B^T_k` after it (with the shared expert's
sigmoid gate where there is one). The matvecs of one input (query / key / value, gate / up, the
Gated DeltaNet projections) are grouped, so their updates add one barrier per group. Gate / up
pairs with an update run as two matvecs and an activation instead of one paired matvec. Chained
adapters are concatenated along the rank (up to 256 per layer on the device; more falls back to
the CPU path).

Routed experts get per-expert side paths (`gpu::decode::Experts::lora`, every expert's `A` and
`B^T` stacked and zero-padded to one rank): with a gate / up update the expert pair kernels write
the raw gate and up outputs, the updates of each routed slot are added and the activation taken
from there; the down projections' updates are added to the output with each slot's routing weight.
In the Metal prefill plan the same runs on the grouped expert rows (each row's expert recorded with
the routing). Prefill without a plan (`OXINFER_PREFILL_GPU=0`, models the plan does not cover)
still runs adapted experts on the CPU, the dense matmuls on the GPU.

The strengths are read on the host before every step and written to the device (`Graph::strengths`
by adapter switch, gain and scale), so `set_lora_strength`, a decision adapter switched per request
or a strength of 0 changes nothing in the plan. At strength 0 the side paths still run (and add
zeros): drop the adapter for the base model's speed.

Inkling-2 has its own Metal engine (docs/inkling.md, "On the GPU") with the same side paths:
the dense updates beside each matmul, the routed experts' down updates per slot, and their gate /
up updates added by the pair kernels before SiTU (prefill passes with those take the per-slot
pair kernels rather than the grouped GEMMs). Its CUDA engine leaves adapted models to the CPU.

## Families and what is adapted

| Family | Layers | Experts | Embedding | Head |
|---|---|---|---|---|
| Qwen3 / 3.5 / 3.6 / 3.8 (dense and MoE), Qwen2 / 2.5-VL, Llama 3, Granite 4, MiniMax, LFM3 (the Qwen code path) | attention `q/k/v/o_proj`, Gated DeltaNet `in_proj_qkv/z/a/b`, `out_proj`, MLP and shared-expert `gate/up/down_proj`; the Qwen3.5 / 3.8 MTP layer (`mtp.fc`, its attention, MLP or experts) | routed experts, per expert | yes | `lm_head` (untied) |
| Gemma 4 / 3 | attention and MLP projections | 26B-A4B routed experts | yes (before the `sqrt(hidden)` scale, as in llama.cpp) | `lm_head` |
| Inkling-2 (`inkling2`, `inkling2_loop`) | `wq_du`, `wk_dv`, `wv_dv`, `wr_du`, `gate_attn`, `attn_sink_proj`, `wo_ud`, the dynamic convolutions' `weight_proj`, the loop gates' `controllers.*` | routed experts (`gate_up_proj`, `down_proj`; interleaved `w13` rows of the release are handled) | yes (before the embedding norm) | `unembed` (as `lm_head`) |

MTP-layer modules (PEFT `mtp.fc`, `mtp.layers.0.*`; GGUF `blk.N.nextn.eh_proj` and the tensors of
the blocks past the main layers) wait in the model until its MTP layer loads (`load_mtp`, when MTP
drafting is on) and are attached there: they change the drafts, not the verified text. Without the
MTP layer they are not used. Qwen4-Exp (Qwen3.8-Flash-Next) MTP layers refuse them, as do the
other families. Other families (Mistral Large 4, DeepSeek V4, GLM-5.3, Kimi, LongCat, Nemotron,
Core ML models) refuse `lora` with an error. An adapter module that matches no layer, or a layer of another shape,
is an error at load: nothing is skipped silently. A looped Inkling layer is adapted once, so its
update applies in every pass, as with the weights it updates (llama.cpp shares them the same way).

## GGUF adapters

Tensors are named after the base file's, in ggml's dimension order (fastest first). This is the
reading of llama.cpp's `src/llama-adapter.cpp` and `build_lora_mm`, `build_lora_mm_id` and the
embedding in `llm_graph_context`:

- a linear layer `W` (`ne [n_in, n_out]`): `X.weight.lora_a` `ne [n_in, r]` and `X.weight.lora_b`
  `ne [r, n_out]`, which are PEFT's `A` `[r, n_in]` and `B` `[n_out, r]` in row-major order, with no
  transposes, applied as `y += s * B (A x)`;
- a stack of experts (`blk.N.ffn_down_exps.weight`, `ffn_gate_up_exps`, `ffn_gate_exps`,
  `ffn_up_exps`; `ne [n_in, n_out, n_expert]`): `lora_a` `ne [n_in, r, n_expert]` and `lora_b`
  `ne [r, n_out, n_expert]`, one pair per expert, applied to the tokens routed to it
  (`ggml_mul_mat_id` over both). A fused `gate_up` pair goes on fused or separate gate / up
  weights;
- `token_embd.weight` (`ne [hidden, vocab]`) has A and B "flipped": `lora_a` `ne [r, vocab]` and
  `lora_b` `ne [r, hidden]`. Token `t`'s row gets `s * B a_t`, where `a_t` is row `t` of `lora_a`.
  The tied output head is not adapted. llama.cpp aborts on such an adapter with a tied head, as
  Llama 3.2 1B has (`GGML_ASSERT(ggml_can_mul_mat(a, b))` in `build_lora_mm` for `output`), and
  runs it as described here on a model with its own head;
- the strength is `s = alpha / r * scale`, where `r` is each pair's `lora_b` `ne[0]`, `alpha` is
  `adapter.lora.alpha` and `scale` is the user's (`:SCALE`, llama.cpp's `--lora-scaled`). Without
  an alpha (absent or 0), `s` is just `scale`. With one rank throughout, `alpha / r` is applied as
  the PEFT path applies its scale. With mixed ranks, each pair's `alpha / r` is folded into its `B`;
- `*_norm.weight` tensors (which `mergekit-extract-lora` adds) are skipped, as llama.cpp skips
  them. MTP-layer tensors (`blk.N.nextn.eh_proj`, blocks past `block_count - nextn_predict_layers`)
  go on the MTP layer (see above). Any other tensor is an error, as are activated LoRAs (`adapter.alora.invocation_tokens`)
  and adapters of another architecture than the model's (`general.architecture`, as in llama.cpp).

Names map to the models' module paths through the tables the GGUF loader uses
(`gguf::hf_tensor`): `blk.N.attn_output.weight` is `layers.N.self_attn.o_proj` (Qwen, Llama,
Gemma) or `layers.N.attn.wo_ud` (Inkling-2), `output.weight` is `lm_head`, and so on. The
converter runs the adapter through the base model's tensor transforms: the Llama query / key row
permutation, and Qwen3.5's tiled value heads. So its pairs are in the base file's layout and go on
as they are. A PEFT adapter on the same GGUF base gets its rows reordered to match. A GGUF adapter
therefore needs a GGUF base of its architecture. On a safetensors checkpoint it is refused: use
the PEFT form there.

AtomicChat's Inkling-2 adapter writes its alpha as `adapter.lora_alpha`, a key llama.cpp does not
read. Both engines then run it without an alpha, at strength 1. With rank 1 and alpha 1 that is
the same scale anyway.

## Merging (`lora_merge`)

With `--lora-merge` (`lora_merge=1`), each update is added to its weight at load
(`Lin::merge_lora`). The weight is decoded to f32, updated and stored again:

| Base weight | Stored as | Precision |
|---|---|---|
| bf16, GGUF BF16 | bf16 | the update rounded with the weight (bf16) |
| GGUF F16 / F32 | F16 / F32 | the same, in f16 / f32 |
| oxinfer Q8 / Q4 (`--quant`) | Q8 / Q4 again | re-quantized: the rounding error of a fresh quantization |
| GGUF block formats: Q4_0 / Q4_1 / Q5_0 / Q5_1 / Q8_0, Q2_K to Q6_K, IQ2_XXS / IQ2_XS / IQ2_S, IQ3_XXS / IQ3_S, IQ4_NL / IQ4_XS, TQ1_0 / TQ2_0, Q1_0 / Q2_0, MXFP4 | their own type, same size | re-quantized with `ggml_quant` (llama.cpp's quantizers, no importance matrix): the rounding error of a fresh quantization at that type |
| other GGUF block formats (IQ1_S / IQ1_M, NVFP4, FP8) | oxinfer Q8 (int8 per 32) | re-quantized at 8 bits |

The embedding's update always stays a runtime one. It costs `r * hidden` per token looked up.
Hadamard-folded weights also stay runtime. Merged updates cost nothing per token, but their
strength is fixed (`set_lora_strength` refuses them), and they carry a fresh quantization's
rounding. The merged tensors are new allocations: on a default load (the file copied to memory) the
originals stay in that copy, so memory grows by the merged tensors' size; with `--mmap` the
originals are reclaimable file pages, and a merged model costs about what an unmerged one does.
Use `--mmap` with `--lora-merge`.

Inkling-2-Mini AD-IQ2_M with AtomicChat's adapter (3096 updates, every routed expert of
`ffn_down_exps`): 9.0 GB peak with the runtime adapter, 12.4 GB merged on a default load (18.7 GB
when merges were stored as Q8), 3.8 GB footprint merged with `--mmap`. KL against the runtime
adapter on five raw prompts 1.6e-3 to 8.9e-3, same top 1 throughout; AtomicChat's baked file (the
same merge, re-quantized by llama.cpp) is 2.6e-3 to 1.3e-2 from it, so the merge is as close to
the runtime adapter as llama.cpp's. The Q8 merge was closer (3e-4 to 1.2e-3) at about twice the
memory. Q2_0 (IQ2_M's experts) is re-quantized with a least-squares fit rather than llama.cpp's
absmax rounding, which never uses the `+2d` level that 6% of those weights take (KL 0.13 to 0.18
that way).

The quantizers (`oxinfer::ggml_quant`: `quantize_row`, `quantize_matrix`, with or without an
importance matrix) give llama.cpp's bytes: identical blocks for every type on 352 rows (random
rows with outliers and Llama-3.2-1B weights, with and without a random importance matrix) against
`ggml_quantize_chunk` built with `-ffp-contract=off`; against the stock arm64 build, whose clang
contracts multiply-adds into FMAs, 97.8 to 100% of blocks are identical (near ties) with the same
RMSE to 1e-4. Departures, for inputs llama.cpp cannot take: IQ2_XXS / IQ2_XS without an importance
matrix use all ones (llama.cpp aborts), and IQ3_S rows with an all-zero group of 32 keep their
index and sign pointers moving (llama.cpp's encoder corrupts the following groups there).
`examples/ggml_quant_check.rs` compares against llama.cpp's output. `--save-quantized` still
writes oxinfer's Q8 / Q4 safetensors, not GGUF.

## Verification

**(a) PEFT and GGUF give the same logits.** A random rank-8 PEFT adapter (alpha 12; `q/k/v/o`,
`gate/up/down` of all 16 layers and `embed_tokens`) for Llama-3.2-1B-Instruct was converted with
llama.cpp's `convert_lora_to_gguf.py --outtype f32`. Strength 0.7, CPU, last-token logits of a raw
prompt; the adapter moves them by up to 3.2 (KL 0.22 against the base):

| Base | PEFT folder vs GGUF adapter |
|---|---|
| bartowski F16 GGUF | bit-identical (max diff 0) |
| bartowski Q4_K_M GGUF | bit-identical |
| unsloth bf16 safetensors + PEFT vs F16 GGUF + GGUF adapter | max diff 3.3e-5, KL 1.7e-11 (as the two bases without an adapter) |
| F16 GGUF, runtime vs `--lora-merge` | KL 3.1e-5 (the merged weights re-rounded to f16) |
| Q4_K_M GGUF, runtime vs `--lora-merge` | KL 1.2e-2 to 3.9e-2 on three prompts (re-quantized as Q4_K; 2.1e-3 when merges were stored as Q8. The base Q4_K_M is KL 6.6e-2 from F16) |

The same adapter without the embedding update was run against llama.cpp's `llama-completion
--lora-scaled FILE:0.7` on the F16 file. Three raw prompts, 48 greedy tokens: identical text in
3/3. llama.cpp aborts on the adapter with the embedding update for this tied-head model (see
above).

**(b) Inkling-2-Mini: AD-IQ2_M plus AtomicChat's adapter, against the baked Uncensored AD-IQ2_M.**
The two IQ2_M files differ in exactly the adapter's 49 tensors. The difference of their
dequantized weights lines up with the adapter's `B A` at a best-fit scale of 1.000 on the Q8_0
`attn_output` tensors (cosine 0.95; the rest is the baked file's re-quantization). On the
routed experts the best-fit scale is 0.85 to 0.89, with cosine 0.14 to 0.33: the experts are Q2_0
and IQ4_NL, whose re-quantization moves them by 3 to 6 times the update. On the Q4_K embedding it
is 0.98. This checks the layout, orientation and scale of each kind of site. Logits at the end of
five neutral raw prompts (CPU):

| Prompt | KL(base + adapter, baked) | KL(base, baked) | KL(base, base + adapter) | same top 1 |
|---|---|---|---|---|
| The sky appears blue during the day because | 2.6e-3 | 8.7e-3 | 5.3e-3 | 3/3 |
| Autumn leaves change color when the weather gets colder, and | 1.3e-2 | 1.5e-2 | 1.7e-2 | 3/3 |
| The capital of Australia is Canberra, a city that | 6.0e-3 | 1.7e-2 | 2.0e-2 | 3/3 |
| A bicycle stays upright while moving mainly because of | 9.7e-3 | 4.3e-2 | 4.5e-2 | 3/3 |
| To grow tomatoes at home, the first thing you should do is | 3.3e-3 | 6.2e-3 | 6.0e-3 | 3/3 |

The adapted base is closer to the baked file than the plain base on every prompt, but not equal.
On neutral text the re-quantization of 49 low-bit tensors moves the logits about as much as the
rank-1 update does. Greedy text, 48 tokens: the adapted base follows the baked file's wording
where the plain base goes elsewhere. On the bicycle prompt it is identical to the baked file to
the end. On the Canberra prompt they share the first 14 words, where the base differs from the
first word. On the sky prompt both start "of the scattering of", where the base starts a quiz. On
the autumn and tomato prompts all three share long prefixes and then split at near ties. At the
end of chat prompts, all three put probability ~1 on the reply's first token (KL < 2e-10). Decoding
on the CPU: 10.5 tok/s with the adapter, 11.8 without (128 tokens, a loaded machine).

**(c) Against llama.cpp on the same files.** Run with `llama-completion --lora`, the same IQ2_M
base and adapter, 48 greedy tokens on the five raw prompts. oxinfer and llama.cpp give identical
text on 3/5 (sky, bicycle, tomatoes). The other two split at near ties after 14 words (Canberra:
"for" / "to") and 31 words (autumn). On the baked file without an adapter, the two engines agree
to the end on 1/5 (Canberra). They split on the other four after 4 to 25 words. So with the
adapter they agree at least as closely as they do on a plain file of this quantization.

**(d) The MTP layer.** A random rank-8 PEFT adapter of Qwen3.5-0.8B's MTP layer only (`mtp.fc`,
its `q/k/v/o_proj`, `gate/up/down_proj`), with MTP drafting (3 tokens, greedy): the text is
identical to plain decoding in every case (drafts are verified), and the drafts follow the adapter.
With `B` of std 0.3 acceptance falls from 69% to 0%, with std 0.01 to 63%; `--lora-merge` (bf16)
gives the same acceptance and per-round histogram as the runtime adapter.

**(e) GPU against CPU.** `OXINFER_GPU_DECODE_CHECK=1` runs every GPU decode step on the CPU too
(64 greedy tokens of a raw prompt, `--no-lookup`):

| Model, adapter | max KL per step | greedy text | without the adapter |
|---|---|---|---|
| Llama-3.2-1B F16 GGUF, GGUF adapter of (a) at 0.7 (113 updates) | 9e-12 | identical | 1.3e-11 |
| Qwen3.5-0.8B bf16, rank 8 on attention, Gated DeltaNet and MLPs | 6e-12 | identical | 6.6e-12 |
| Gemma 4 12B bf16, rank 16 on the MLPs, strength 0.5 | 1.8e-9 | identical (40 tokens) | |
| Llama-3.2-1B Q4_K_M | 9e-4 | identical | 9.5e-4 (the CPU's int8 activations) |
| Qwen3.5-35B-A3B oxinfer Q4, rank 4 on every expert of 5 layers, shared experts, attention (2298 updates) | 4.8e-3, same argmax at every step | splits at a near tie after 9 lines | 2.9e-3 |

Prefill plan, last-token distribution after a 2358-token prompt (Qwen3.5-0.8B bf16, the adapter
above at 0.3): KL 2.8e-7 against the CPU (8.8e-8 without it), 0.21 s against 0.18 s. The routed
experts' side paths in exact arithmetic are checked by `tests/gemma_moe_gpu.rs`: a tiny Gemma 4
MoE with an adapter on its attention, dense MLP and every expert (gate / up only, down only, and
both), prefill plan and decode steps against the CPU, at two strengths on the same plans (the
same error as without the adapter: 2.4e-3 prefill and 1.6e-3 decode in bf16; strengths 1 and 0.4
are further apart than 0.1).

Decode speed on Metal, tok/s (M5 Max, a loaded machine, so +-10%; 128 to 256 tokens):

| Model, adapter | base, GPU | adapter, GPU | adapter, CPU decode (before) |
|---|---|---|---|
| Llama-3.2-1B Q4_K_M, rank 8, 113 updates | 300-360 | 235-310 | 80-180 |
| Qwen3-8B Q4_K_M, rank 16 on all 252 projections | 80-87 | 61-73 | 12-35 |
| Qwen3.5-35B-A3B oxinfer Q4, 2298 updates (experts of 5 layers) | 155 | 139 | 60 |

The cost is mostly the extra barrier after each group of adapted matvecs and the `B` reads (the
adapter's f32 factors are read once per token: 175 MB for the Qwen3-8B adapter).

Unit tests: `moe::tests::expert_lora_matches_merged_weights` checks expert updates against
weights merged in f64 for the three expert layouts: PEFT stacked, PEFT per expert, and the
per-expert fused `gate_up_proj` of GGUF stacks. It covers fused and split storage, and decode,
short passes and grouped prefill. `lora::tests` checks the low-rank product and the name mapping.
