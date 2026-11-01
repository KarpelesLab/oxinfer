#!/usr/bin/env python3
# Builds a tiny random Mistral Large 4 checkpoint in the official layout (params.json +
# consolidated.safetensors with Mistral-native names, config.json with transformers_weights),
# with every architectural feature of the real model, and checks transformers loads it back.
#
# usage: mistral4_tiny.py OUT_DIR REAL_REPO_DIR
#   REAL_REPO_DIR holds tekken.json, tokenizer.json, chat_template.jinja, tokenizer_config.json and
#   generation_config.json of mistralai/Mistral-Large-4-1T-A52B (small files).
# Then: ref_mistral4.py OUT_DIR ref.safetensors 300 16, and
#   cargo run --release --example mistral4_check -- OUT_DIR ref.safetensors [--chunk N]
# llama.cpp's `convert_hf_to_gguf.py --mistral-format OUT_DIR` gives the GGUF to check the same way.
import json, math, os, shutil, sys

import torch
from safetensors.torch import save_file

from transformers import MistralLarge4Config, MistralLarge4ForConditionalGeneration
from transformers.models.mistral_large4.convert_mistral_large4_to_hf import convert_config

out = sys.argv[1]
hf = sys.argv[2]  # folder with tekken.json / tokenizer.json / chat_template.jinja of the real repo
os.makedirs(out, exist_ok=True)
torch.manual_seed(1234)

n_layers = 8
params = {
    "dim": 256, "n_layers": n_layers, "head_dim": 128, "hidden_dim": 512, "n_heads": 4, "n_kv_heads": 4,
    "rope_theta": 10000.0, "norm_eps": 1e-05, "vocab_size": 204800, "tied_embeddings": False,
    "max_position_embeddings": 1048576, "ragged_attention": None, "nope": "rope,rope,rope,nope",
    "softmax_tanh_gating": 30.0, "llama_4_scaling": None, "q_lora_rank": None, "qk_rope_head_dim": 64,
    "qk_nope_head_dim": 64, "kv_lora_rank": 256, "v_head_dim": 128, "ssmax_scaling": True, "lerope": True,
    "dsa": {"topk": 16, "n_heads": 16, "head_dim": 128, "rope_head_dim": 64, "match_nope": True,
            "ssmax_scaling": True, "norm_eps": 1e-06,
            "sharing": "full,full,shared,full,full,shared,full,shared", "fp8_indexer": True, "fp8_kv_cache": True},
    "yarn": {"original_max_position_embeddings": 512, "factor": 2048, "apply_scale": False, "beta": 32, "alpha": 1},
    "kv_sharing": {"source_layers_str": "0,1,2,3,4,5,6,3"},
    "moe": {"expert_parallel": 1, "expert_model_parallel": 1, "route_every_n": 1, "first_k_dense_replace": 1,
            "num_experts": 16, "num_experts_per_tok": 4, "num_expert_groups": 1, "num_expert_groups_per_tok": 1,
            "routed_scale": 1.0, "expert_hidden_dim": 256, "num_shared_experts": 1},
    "vision_encoder": {"image_token_id": 10, "image_break_token_id": 12, "image_end_token_id": 13,
                       "intermediate_size": 256, "num_hidden_layers": 2, "num_attention_heads": 2,
                       "mm_projector_id": "patch_merge", "spatial_merge_size": 2, "hidden_size": 128,
                       "num_channels": 3, "image_size": 1540, "max_image_size": 1540, "patch_size": 14,
                       "rope_theta": 10000.0, "add_pre_mm_projector_layer_norm": True, "adapter_bias": False},
}
cfgd = convert_config(params, {"bos_token_id": 1, "eos_token_id": 2})
cfg = MistralLarge4Config(**cfgd)
model = MistralLarge4ForConditionalGeneration(cfg).float().eval()

g = torch.Generator().manual_seed(42)


def rn(*shape):
    return torch.randn(*shape, generator=g)


with torch.no_grad():
    for name, p in model.named_parameters():
        if name.endswith("softmax_scaling.scale"):
            p.copy_(0.1 + 0.3 * torch.rand(p.shape, generator=g))
        elif name.endswith("softmax_scaling.bias"):
            p.copy_(0.6 + 0.8 * torch.rand(p.shape, generator=g))
        elif "norm" in name and name.endswith("weight"):
            p.copy_(1.0 + 0.2 * rn(*p.shape))
        elif name.endswith("bias"):
            p.copy_(0.1 * rn(*p.shape))
        elif "embed_tokens" in name:
            p.copy_(rn(*p.shape))
        elif p.ndim >= 2:
            fan_in = p.shape[-1] if p.ndim != 4 else p[0].numel()
            gain = 1.0
            if "q_proj" in name or "kv_a_proj" in name or "wq_b" in name or "indexer.wk" in name:
                gain = 2.0
            if "mlp.gate.weight" in name:
                gain = 3.0
            if "lm_head" in name:
                gain = 4.0
            p.copy_(gain * rn(*p.shape) / math.sqrt(fan_in))
        else:
            p.copy_(0.1 * rn(*p.shape))
    # LeRoPE alpha is a buffer (persistent), not a parameter.
    model.model.language_model.rotary_emb.alpha.copy_(0.3 * rn(*model.model.language_model.rotary_emb.alpha.shape))
    for name, p in list(model.named_parameters()) + [("alpha", model.model.language_model.rotary_emb.alpha)]:
        p.copy_(p.bfloat16().float())

sd = model.state_dict()
tc = cfg.text_config
nat = {}


def put(k, v):
    nat[k] = v.detach().clone().contiguous()


def permute_inv(w, n_heads):
    # native interleaved rows (2i, 2i+1) <- rotate-half rows (i, half + i): inverse of PermuteForRope
    dim0 = w.shape[0]
    half = dim0 // n_heads // 2
    return w.view(n_heads, 2, half, w.shape[1]).transpose(1, 2).reshape(dim0, w.shape[1])


vh = cfg.vision_config.num_attention_heads
mi = tc.moe_intermediate_size
for k, v in sd.items():
    if k == "model.language_model.rotary_emb.alpha":
        put("lerope.alpha", v)
        continue
    if k.startswith("model.language_model.rotary_emb"):
        continue
    if k.startswith("model.language_model."):
        r = k[len("model.language_model."):]
        if r == "embed_tokens.weight":
            put("tok_embeddings.weight", v)
            continue
        if r == "norm.weight":
            put("norm.weight", v)
            continue
        assert r.startswith("layers."), r
        _, i, rest = r.split(".", 2)
        p = f"layers.{i}."
        if rest == "mlp.experts.gate_up_proj":
            for e in range(v.shape[0]):
                put(f"{p}experts.{e}.w1.weight", v[e, :mi])
                put(f"{p}experts.{e}.w3.weight", v[e, mi:])
            continue
        if rest == "mlp.experts.down_proj":
            for e in range(v.shape[0]):
                put(f"{p}experts.{e}.w2.weight", v[e])
            continue
        m = {
            "input_layernorm.weight": "attention_norm.weight",
            "post_attention_layernorm.weight": "ffn_norm.weight",
            "self_attn.q_proj.weight": "attention.wq.weight",
            "self_attn.kv_a_proj_with_mqa.weight": "attention.wkv_a_with_mqa.weight",
            "self_attn.kv_a_layernorm.weight": "attention.kv_a_norm.weight",
            "self_attn.kv_b_proj.weight": "attention.wkv_b.weight",
            "self_attn.o_proj.weight": "attention.wo.weight",
            "self_attn.softmax_scaling.scale": "attention.ssmax_scaling.s",
            "self_attn.softmax_scaling.bias": "attention.ssmax_scaling.b",
            "self_attn.indexer.softmax_scaling.scale": "attention.indexer.ssmax_scaling.s",
            "self_attn.indexer.softmax_scaling.bias": "attention.indexer.ssmax_scaling.b",
            "mlp.gate_proj.weight": "feed_forward.w1.weight",
            "mlp.down_proj.weight": "feed_forward.w2.weight",
            "mlp.up_proj.weight": "feed_forward.w3.weight",
            "mlp.gate.weight": "gate.weight",
            "mlp.shared_experts.gate_proj.weight": "shared_experts.w1.weight",
            "mlp.shared_experts.down_proj.weight": "shared_experts.w2.weight",
            "mlp.shared_experts.up_proj.weight": "shared_experts.w3.weight",
        }
        if rest in m:
            put(p + m[rest], v)
        elif rest.startswith("self_attn.indexer."):
            put(p + "attention.indexer." + rest[len("self_attn.indexer."):], v)
        else:
            raise SystemExit(f"unmapped {k}")
        continue
    if k == "lm_head.weight":
        put("output.weight", v)
        continue
    if k.startswith("model.multi_modal_projector."):
        r = k[len("model.multi_modal_projector."):]
        m = {"norm.weight": "pre_mm_projector_norm.weight",
             "patch_merger.merging_layer.weight": "patch_merger.merging_layer.weight",
             "linear_1.weight": "vision_language_adapter.w_in.weight",
             "linear_2.weight": "vision_language_adapter.w_out.weight"}
        put(m[r], v)
        continue
    if k.startswith("model.vision_tower."):
        r = k[len("model.vision_tower."):]
        r = (r.replace(".feed_forward.gate_proj.", ".feed_forward.w1.")
             .replace(".feed_forward.down_proj.", ".feed_forward.w2.")
             .replace(".feed_forward.up_proj.", ".feed_forward.w3.")
             .replace(".attention.v_proj.", ".attention.wv.")
             .replace(".attention.o_proj.", ".attention.wo."))
        if ".attention.q_proj." in r:
            put("vision_encoder." + r.replace(".q_proj.", ".wq."), permute_inv(v, vh))
            continue
        if ".attention.k_proj." in r:
            put("vision_encoder." + r.replace(".k_proj.", ".wk."), permute_inv(v, vh))
            continue
        put("vision_encoder." + r, v)
        continue
    raise SystemExit(f"unmapped {k}")

nat = {k: v.bfloat16() for k, v in nat.items()}
save_file(nat, os.path.join(out, "consolidated.safetensors"), metadata={"format": "pt"})
json.dump({"metadata": {"total_size": sum(v.numel() * 2 for v in nat.values())},
           "weight_map": {k: "consolidated.safetensors" for k in sorted(nat)}},
          open(os.path.join(out, "consolidated.safetensors.index.json"), "w"), indent=2)
json.dump(params, open(os.path.join(out, "params.json"), "w"), indent=2)
cfg.to_json_file(os.path.join(out, "config.json"))
c = json.load(open(os.path.join(out, "config.json")))
c["transformers_weights"] = "consolidated.safetensors.index.json"
json.dump(c, open(os.path.join(out, "config.json"), "w"), indent=2)
for f in ["tekken.json", "tokenizer.json", "chat_template.jinja", "tokenizer_config.json", "generation_config.json"]:
    shutil.copy(os.path.join(hf, f), os.path.join(out, f))

# Round trip: transformers loads the native layout back to the same parameters.
m2 = MistralLarge4ForConditionalGeneration.from_pretrained(out, dtype=torch.float32)
sd2 = m2.state_dict()
bad = 0
for k, v in sd.items():
    if "rotary_emb.inv_freq" in k or "original_inv_freq" in k:
        continue
    if k not in sd2:
        print("missing after reload", k)
        bad += 1
        continue
    d = (sd2[k].float() - v.float()).abs().max().item()
    if d > 0:
        print("diff", k, d)
        bad += 1
print("reload mismatches:", bad, "tensors:", len(nat))
torch.save(model.state_dict(), os.path.join(out, "ref_state.pt"))
