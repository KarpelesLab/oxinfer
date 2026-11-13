#!/usr/bin/env python3
# Builds a tiny random Kimi K3.1 checkpoint in the layout of moonshotai/Kimi-K3.1-Pro (config.json
# with its keys, `model.safetensors` with its tensor names under `language_model.`), with every
# feature of the text model: KDA layers (per-channel decay gate with a lower bound, full-rank
# output gate, `A_log` in a zero-padded `[head_dim]` buffer as released), gated MLA layers with a
# lightning indexer whose top-k is small enough that most queries are sparse, cross-layer
# attention residuals over several blocks, a dense first layer, latent MoE with its norm, shared
# experts, the SiTU activation with both betas. The module tree and the parameter shapes are those
# of Moonshot's `modeling_kimi_linear.py` (tools/ref_kimi_k31.py).
#
# usage: kimi_k31_tiny.py OUT_DIR CODE_DIR [--mxfp4] [--k3] [--smooth]
#   CODE_DIR: `modeling_kimi_linear.py` of moonshotai/Kimi-K3 and the small files of
#   moonshotai/Kimi-K3.1-Pro (configuration_kimi_k3.py; tiktoken.model and tokenizer_config.json
#   are copied when present). Matrices are stored in bf16, as released (oxinfer keeps most of
#   them in bf16: float32 weights would be rounded, and a random model with hard choices, the
#   experts and the keys of each token, amplifies that into different choices).
#   --mxfp4: the routed experts as compressed-tensors `mxfp4-pack-quantized` (`weight_packed` /
#   `weight_scale`), as released. --k3: without the indexers (Kimi K3: dense MLA).
#   --smooth: no hard choice (every expert routed, every key attended), so that an approximate
#   kernel (oxinfer runs 4-bit weights against int8 activations) shows as a small difference
#   instead of as other experts chosen: use it with --mxfp4.
# Then: ref_kimi_k31.py OUT_DIR CODE_DIR ref.safetensors 96 16, and
#   cargo run --release --example kimi_check -- OUT_DIR ref.safetensors [--chunk N]
import json, math, os, shutil, sys

import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ref_kimi_k31 import build, reference, unpack_mxfp4

out, code = sys.argv[1], sys.argv[2]
mxfp4, k3, smooth = "--mxfp4" in sys.argv, "--k3" in sys.argv, "--smooth" in sys.argv
os.makedirs(out, exist_ok=True)
for fn in ("tiktoken.model", "tokenizer_config.json"):
    if os.path.exists(os.path.join(code, fn)):
        shutil.copy(os.path.join(code, fn), os.path.join(out, fn))

n_layers = 13
full = [4, 8, 12, 13]  # 1-based, as in the release: every fourth layer and the last
text = {
    "model_type": "kimi_linear", "architectures": ["KimiLinearForCausalLM"], "vocab_size": 163840,
    "hidden_size": 64, "intermediate_size": 160, "num_hidden_layers": n_layers,
    "num_attention_heads": 4, "num_key_value_heads": 4, "hidden_act": "situ",
    "activation_situ_beta": 4.0, "activation_situ_linear_beta": 25.0, "rms_norm_eps": 1e-05,
    "attn_res_block_size": 4, "first_k_dense_replace": 1, "moe_layer_freq": 1,
    "num_experts": 8, "num_experts_per_token": 3, "num_shared_experts": 2, "moe_intermediate_size": 64,
    "routed_expert_hidden_size": 32, "latent_moe_use_norm": True, "moe_renormalize": True,
    "moe_router_activation_func": "sigmoid", "routed_scaling_factor": 1.0, "num_expert_group": 1,
    "topk_group": 1, "topk_method": "noaux_tc", "use_grouped_topk": True,
    "q_lora_rank": 24, "kv_lora_rank": 16, "qk_nope_head_dim": 16, "qk_rope_head_dim": 8, "v_head_dim": 16,
    "mla_use_nope": True, "mla_use_output_gate": True,
    "linear_attn_config": {
        "full_attn_layers": full, "kda_layers": [l for l in range(1, n_layers + 1) if l not in full],
        "head_dim": 16, "num_heads": 4, "short_conv_kernel_size": 4, "gate_lower_bound": -5.0,
        "use_full_rank_gate": True,
    },
    "max_position_embeddings": 1048576, "num_nextn_predict_layers": 0, "tie_word_embeddings": False,
    "bos_token_id": 163584, "eos_token_id": 163586, "pad_token_id": 163839,
    "dtype": "bfloat16",
}
if smooth:
    text["num_experts_per_token"] = text["num_experts"]
if not k3:
    text["dsa_config"] = {
        "dsa_layers": full, "index_topk": 4096 if smooth else 12, "index_head_dim": 16, "index_num_heads": 16, "index_rope_dim": 0,
        "index_k_norm_type": "layernorm", "index_topk_freq": 1, "index_skip_topk_offset": 1,
    }
if mxfp4:
    text["quantization_config"] = {
        "quant_method": "compressed-tensors", "format": "mxfp4-pack-quantized",
        "config_groups": {"group_0": {"format": "mxfp4-pack-quantized", "targets": ["Linear"],
                                      "weights": {"num_bits": 4, "group_size": 32, "type": "float"}}},
    }
cfg = {
    "architectures": ["KimiK3ForConditionalGeneration" if k3 else "KimiK31ForConditionalGeneration"],
    "model_type": "kimi_k3" if k3 else "kimi_k31", "dtype": "bfloat16",
    "bos_token_id": 163584, "eos_token_id": 163586, "pad_token_id": 163839,
    "media_placeholder_token_id": 163605, "tie_word_embeddings": False, "text_config": text,
}

torch.set_default_dtype(torch.float32)
M = reference(code)
model = build(M, cfg)
g = torch.Generator().manual_seed(31)
heads, head_dim = text["linear_attn_config"]["num_heads"], text["linear_attn_config"]["head_dim"]


def rn(*shape):
    return torch.randn(*shape, generator=g)


# Gains near one keep the model well conditioned: with larger ones it amplifies float32 rounding
# until the hard choices (experts, keys) of some token differ, in the reference itself between a
# single pass and one-token passes.
GAIN = {"lm_head.weight": 2.0, "o_proj.weight": 0.5, "down_proj.weight": 0.5, "w2.weight": 0.5,
        "routed_expert_up_proj.weight": 0.5, "wq_b.weight": 2.0, "wk.weight": 2.0}
tensors = {}
with torch.no_grad():
    for name, p in model.named_parameters():
        last = name.split(".")[-1]
        key = "language_model." + name
        if last == "A_log":
            # Per head, in the first `num_heads` entries of a `[head_dim]` buffer (zero after).
            v = torch.zeros(head_dim)
            v[:heads] = 0.6 * rn(heads)
        elif last == "dt_bias":
            v = 0.5 * rn(*p.shape)
        elif last == "e_score_correction_bias":
            v = 0.3 * rn(*p.shape)
        elif name.endswith("_res_proj.weight"):
            v = rn(*p.shape) / math.sqrt(p.shape[-1])
        elif "conv1d" in name:
            v = 0.7 * rn(*p.shape)
        elif "norm" in name and last == "weight":
            v = 1.0 + 0.2 * rn(*p.shape)
        elif "norm" in name and last == "bias":
            v = 0.2 * rn(*p.shape)
        elif name == "model.embed_tokens.weight":
            v = rn(*p.shape)
        elif p.ndim >= 2:
            gain = next((gv for k, gv in GAIN.items() if name.endswith(k)), 1.0)
            v = gain * rn(*p.shape) / math.sqrt(p.shape[-1])
        else:
            v = 0.1 * rn(*p.shape)
        if mxfp4 and ".block_sparse_moe.experts." in name:
            # Random codes, scales around the weights' size.
            rows, cols = p.shape
            packed = torch.randint(0, 256, (rows, cols // 2), generator=g, dtype=torch.uint8)
            e = round(127 + math.log2(1.0 / math.sqrt(cols)) - 1)
            scale = torch.randint(e - 1, e + 2, (rows, cols // 32), generator=g, dtype=torch.uint8)
            assert unpack_mxfp4(packed, scale).shape == p.shape
            tensors[key + "_packed"], tensors[key + "_scale"] = packed, scale
            continue
        # As released: the convolutions, A_log, dt_bias and the router bias stay in float32.
        keep = last in ("A_log", "dt_bias", "e_score_correction_bias") or "conv1d" in name
        if not keep:
            v = v.to(torch.bfloat16)
        tensors[key] = v.contiguous()

save_file(tensors, os.path.join(out, "model.safetensors"), metadata={"format": "pt"})
with open(os.path.join(out, "config.json"), "w") as f:
    json.dump(cfg, f, indent=2)
print("saved", out, len(tensors), "tensors")
