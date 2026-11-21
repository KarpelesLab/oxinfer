# LoRA adapters for language models

oxinfer applies LoRA adapters to the language models at run time without merging them: the base
weights stay as they were loaded (bf16, Q8 / Q4, MLX, any GGUF block format), and each adapted
layer adds `strength * scale * B (A x)` to its output. The low-rank products run on the CPU in f32,
after the base matmul. An adapter at strength 0 runs exactly the base model's code path. Two
formats are read, for the same families:

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
| Server | the same model options: `qwen=Qwen/Qwen3-8B,lora=./adapter:0.5,lora=./other`; in a JSON config, `"options": {"lora": ["./a", "./b:0.5"]}` (an array) or one string |

The prompt cache keys snapshots by the adapters and their strengths, so a cached conversation of
the base model is never restored into an adapted one (or the reverse). Models with an adapter
decode without the GPU decode plan and run their experts on the CPU in prefill while it is on
(the device runs those in one submission, with no place for the side path); dense prefill matmuls
stay on the GPU. A strength set to 0 keeps that cost; drop the adapter to get it back.

## Families and what is adapted

| Family | Layers | Experts | Embedding | Head |
|---|---|---|---|---|
| Qwen3 / 3.5 / 3.6 / 3.8 (dense and MoE), Qwen2 / 2.5-VL, Llama 3, Granite 4, MiniMax, LFM3 (the Qwen code path) | attention `q/k/v/o_proj`, Gated DeltaNet `in_proj_qkv/z/a/b`, `out_proj`, MLP and shared-expert `gate/up/down_proj` | routed experts, per expert | yes | `lm_head` (untied) |
| Gemma 4 / 3 | attention and MLP projections | 26B-A4B routed experts | yes (before the `sqrt(hidden)` scale, as in llama.cpp) | `lm_head` |
| Inkling-2 (`inkling2`, `inkling2_loop`) | `wq_du`, `wk_dv`, `wv_dv`, `wr_du`, `gate_attn`, `attn_sink_proj`, `wo_ud`, the dynamic convolutions' `weight_proj`, the loop gates' `controllers.*` | routed experts (`gate_up_proj`, `down_proj`; interleaved `w13` rows of the release are handled) | yes (before the embedding norm) | `unembed` (as `lm_head`) |

Other families (Mistral Large 4, DeepSeek V4, GLM-5.3, Kimi, LongCat, Nemotron, Core ML models)
refuse `lora` with an error. An adapter module that matches no layer, or a layer of another shape,
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
  them. Any other tensor is an error, as are activated LoRAs (`adapter.alora.invocation_tokens`)
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
| GGUF block formats (Q4_K, IQ2_M's Q2_0, IQ4_NL, ...) | oxinfer Q8 (int8 per 32) | re-quantized at 8 bits: close to the float merge, but more memory than the low-bit original |

The embedding's update always stays a runtime one. It costs `r * hidden` per token looked up.
Hadamard-folded weights also stay runtime. Merged updates cost nothing per token, but their
strength is fixed (`set_lora_strength` refuses them), and the memory grows wherever low-bit weights
become Q8. Inkling-2-Mini AD-IQ2_M with AtomicChat's adapter merged peaks at 18.7 GB resident,
against about 9 GB unmerged. That adapter touches every routed expert of `ffn_down_exps`. Its
logits are within KL 3e-4 to 1.2e-3 of the runtime adapter's. Runtime adapters are the default
because a merge into the original low-bit format cannot be redone here: there is no ggml block
quantizer in oxinfer. AtomicChat's baked files are such merges, re-quantized by llama.cpp.

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
| Q4_K_M GGUF, runtime vs `--lora-merge` | KL 2.1e-3 (re-quantized as Q8) |

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

Unit tests: `moe::tests::expert_lora_matches_merged_weights` checks expert updates against
weights merged in f64 for the three expert layouts: PEFT stacked, PEFT per expert, and the
per-expert fused `gate_up_proj` of GGUF stacks. It covers fused and split storage, and decode,
short passes and grouped prefill. `lora::tests` checks the low-rank product and the name mapping.
