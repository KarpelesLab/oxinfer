#!/usr/bin/env python3
# Builds a tiny random DeepSeek V4.1 checkpoint in the official layout (config.json with the keys
# of deepseek-ai/DeepSeek-V4.1-Flash, `model.safetensors` with its tensor names, float instead of
# FP8 / FP4), with every feature of the text model: window-only layers, key source layers at
# rates 2 and 1 with layers reading their keys, index source layers (one of them not a key
# source) with layers reusing their choice, a top-k small enough that most queries are sparse,
# candidate blocks that restrict the last index layer, engram lookups before two layers,
# hyper-connections, attention sinks, grouped output projection, YaRN on the compressed layers,
# clamped SwiGLU experts. The module tree and the parameter shapes are those of DeepSeek's own
# `inference/model.py` (tools/ref_deepseek41.py).
#
# usage: deepseek41_tiny.py OUT_DIR REAL_REPO_DIR [--f32] [--vision]
#   REAL_REPO_DIR holds tokenizer.json / tokenizer_config.json and inference/ of
#   deepseek-ai/DeepSeek-V4.1-Flash. Matrices are stored in bf16 (--f32: float32).
#   --vision: with a small vision tower and aligner (and the image routing bias).
# The engram hash inputs the reference derives from the tokenizer and a NumPy generator are
# stored as `engram.token_map` / `engram.multipliers` (oxinfer reads them there; a GGUF
# conversion carries them in its header).
# Then: ref_deepseek41.py OUT_DIR REAL_REPO_DIR/inference ref.safetensors 300 16, and
#   cargo run --release --example deepseek4_check -- OUT_DIR ref.safetensors [--chunk N]
import json, math, os, shutil, sys

import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ref_deepseek41 import model_args, reference, tokenizer

out, real = sys.argv[1], sys.argv[2]
f32 = "--f32" in sys.argv
vision = "--vision" in sys.argv
os.makedirs(out, exist_ok=True)
for fn in ("tokenizer.json", "tokenizer_config.json"):
    shutil.copy(os.path.join(real, fn), os.path.join(out, fn))
M = reference(os.path.join(real, "inference"))

ratios = [0, 2, 2, 2, 1, 1, 1, 0]
text = {
    "model_type": "deepseek_v41_text", "vocab_size": 129280, "hidden_size": 256, "moe_intermediate_size": 64,
    "num_hidden_layers": len(ratios), "num_attention_heads": 4, "num_key_value_heads": 1, "head_dim": 128,
    "qk_rope_head_dim": 32, "q_lora_rank": 64, "o_lora_rank": 48, "o_groups": 2, "hidden_act": "silu",
    "swiglu_limit": 1.5, "rms_norm_eps": 1e-20, "tie_word_embeddings": False, "max_position_embeddings": 1048576,
    "rope_theta": 10000,
    "rope_scaling": {"rope_type": "yarn", "factor": 4, "beta_fast": 32, "beta_slow": 1,
                     "original_max_position_embeddings": 64},
    "n_routed_experts": 8, "n_shared_experts": 1, "num_experts_per_tok": 2, "scoring_func": "sqrtsoftplus",
    "topk_method": "noaux_tc", "norm_topk_prob": True, "routed_scaling_factor": 1.5, "sliding_window": 16,
    "compress_ratios": ratios, "compress_rope_theta": 160000,
    # Layer 3 runs its own indexer on layer 1's keys; layers 2, 5 reuse a choice; layer 6 chooses
    # inside layer 4's candidate blocks.
    "kv_source_layer_ids": [1, 4], "index_source_layer_ids": [1, 3, 4, 6],
    "index_n_heads": 16, "index_head_dim": 64, "index_topk": 8,
    "candidate_source_layer_id": 4, "candidate_topk_blocks": 6, "candidate_block_size": 4,
    "hc_mult": 4, "hc_sinkhorn_iters": 20, "hc_eps": 1e-06,
    "engram_layer_ids": [1, 4], "engram_num_embeddings": [0, 0], "engram_max_ngram_size": 3,
    "engram_vocab_size": 500, "engram_n_heads": 2, "engram_head_dim": 32, "engram_pad_token_id": 2,
    "engram_compressed_vocab_size": 99092, "num_nextn_predict_layers": 0,
}
cfg = {
    "architectures": ["DeepseekV41ForCausalLM"], "model_type": "deepseek_v41",
    "dtype": "float32" if f32 else "bfloat16", "bos_token_id": 0, "eos_token_id": 1, "pad_token_id": 2,
    "image_token_id": 129264, "text_config": text,
}
if vision:
    cfg["vision_config"] = {
        "model_type": "deepseek_v41_vision", "num_hidden_layers": 3, "hidden_size": 64, "num_attention_heads": 4,
        "intermediate_size": 96, "patch_size": 14, "rope_theta": 10000, "downsample_ratio": 3,
        "max_image_tokens": 64, "min_pixels": 3136, "max_wh_ratio": None,
    }

# The rows of each engram table: its heads' prime bucket counts.
layout = M.EngramLayout.from_args(model_args(M, cfg, 1024))
text["engram_num_embeddings"] = [sum(p for per in layer for p in per) for layer in layout.primes]

torch.set_default_dtype(torch.float32)
tok = tokenizer(out)
model = M.Transformer(model_args(M, cfg, 1024), tok).float()
g = torch.Generator().manual_seed(42)


def rn(*shape):
    return torch.randn(*shape, generator=g)


GAIN = {"wq_a.weight": 2.0, "wq_b.weight": 2.0, "wkv.weight": 2.0, "wgate.weight": 3.0, "gate.weight": 3.0,
        "head.weight": 4.0, "wk.weight": 2.0}
tensors = {}
with torch.no_grad():
    for name, p in model.named_parameters():
        if name.startswith("mtp."):
            continue
        last = name.split(".")[-1]
        if "norm" in name and last == "weight":
            v = 1.0 + 0.2 * rn(*p.shape)
        elif last == "attn_sink":
            v = rn(*p.shape)
        elif last.endswith("_base"):
            v = 0.5 * rn(*p.shape)
        elif last.endswith("_scale"):
            v = 0.5 + torch.rand(p.shape, generator=g)
        elif last.endswith("_fn"):
            v = rn(*p.shape) / math.sqrt(p.shape[-1]) * 4.0
        elif last in ("q_weight", "k_weight"):
            v = 1.0 + 0.3 * rn(*p.shape)
        elif last in ("image_start", "image_end", "image_newline") or name in ("embed.weight",) or ".engram.embed." in name:
            v = rn(*p.shape)
        elif last in ("bias", "bias_vl") and ".gate." in name:
            v = 0.3 * rn(*p.shape)
        elif p.ndim >= 2:
            gain = next((gv for k, gv in GAIN.items() if name.endswith(k)), 1.0)
            v = gain * rn(*p.shape) / math.sqrt(p.shape[-1])
        else:
            v = 0.1 * rn(*p.shape)
        # The reference keeps the hc tensors, norms, sinks and biases in float32.
        if p.ndim >= 2 and not f32 and "hc_" not in last:
            v = v.to(torch.bfloat16)
        tensors[name] = v.contiguous()
if model.engram_hash is not None:
    tensors["engram.token_map"] = model.engram_hash.token_map.to(torch.int32).contiguous()
    tensors["engram.multipliers"] = model.engram_hash.multipliers.to(torch.int64).contiguous()

save_file(tensors, os.path.join(out, "model.safetensors"), metadata={"format": "pt"})
with open(os.path.join(out, "config.json"), "w") as f:
    json.dump(cfg, f, indent=2)
print("saved", out, len(tensors), "tensors;", "engram rows", text["engram_num_embeddings"])
