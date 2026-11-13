#!/usr/bin/env python3
# Builds a tiny random Raon-1 nano checkpoint in the layout of KRAFTON/Raon-1-nano (config.json
# with its keys, `model.safetensors` with its tensor names), with every feature of the model: RDA
# layers (short convolutions, per-channel decay with a lower bound, write gate, gated output
# norm), gated NoPE MLA layers, the Raon_mHC residual streams with their selectors and mixing
# charts, per-layer embeddings, the tied head. The module tree and the parameter shapes are those
# of KRAFTON's `modeling_raon1_nano.py`.
#
# usage: raon_tiny.py OUT_DIR CODE_DIR
#   CODE_DIR: `modeling_raon1_nano.py` and `configuration_raon1_nano.py` of KRAFTON/Raon-1-nano.
#   Matrices are stored in bf16, as released (`A_log`, `dt_bias` and `eta` in float32).
# Then: ref_raon.py OUT_DIR CODE_DIR ref.safetensors 96 16, and
#   cargo run --release --example lm_check -- OUT_DIR ref.safetensors [--chunk N]
import json, math, os, sys

import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ref_raon import reference

out, code = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
n_layers = 6
mla = [4, 6]  # 1-based, as in the release: every fourth layer (and here the last)
cfg = {
    "architectures": ["Raon1NanoForCausalLM"], "model_type": "raon1_nano", "dtype": "bfloat16",
    "hidden_size": 64, "num_hidden_layers": n_layers, "num_attention_heads": 4, "head_dim": 16,
    "vocab_size": 512, "tie_word_embeddings": True, "max_position_embeddings": 262144,
    "rms_norm_eps": 1e-06, "rope_parameters": None, "hidden_act": "silu", "intermediate_size": 160,
    "q_lora_rank": 24, "kv_lora_rank": 16, "qk_nope_head_dim": 16, "qk_rope_head_dim": 8, "v_head_dim": 16,
    "mla_use_nope": True, "mla_use_output_gate": True, "raon_mhc_streams": 4,
    "rda_config": {
        "num_heads": 3, "head_dim": 16, "conv_kernel_size": 4, "full_rank_gate": True, "gate_lower_bound": -5.0,
        "mla_layers": mla, "rda_layers": [l for l in range(1, n_layers + 1) if l not in mla],
    },
    "rda_decay_rank": 12, "per_layer_embedding_dim": 8, "per_layer_embedding_layout_version": 2,
    "pad_token_id": 0, "eos_token_id": [1],
}
M = reference(code)
C = sys.modules["raon_ref.configuration_raon1_nano"]
skip = ("architectures", "model_type", "dtype")
model = M.Raon1NanoForCausalLM(C.Raon1NanoConfig(**{k: v for k, v in cfg.items() if k not in skip}))
g = torch.Generator().manual_seed(7)


def rn(*shape):
    return torch.randn(tuple(shape), generator=g)


GAIN = {"o_proj.weight": 0.5, "down_proj.weight": 0.5}
tensors = {}
with torch.no_grad():
    for name, p in model.named_parameters():
        last = name.split(".")[-1]
        if name == "lm_head.weight":
            continue  # tied
        if last == "A_log":
            v = 0.6 * rn(*p.shape)
        elif last == "dt_bias":
            v = 0.5 * rn(*p.shape)
        elif last == "eta":
            v = rn(*p.shape)
        elif last.startswith("alpha_"):
            v = 1.0 + 0.3 * rn(*p.shape)
        elif last.startswith("b_") and p.ndim == 1:
            v = 0.3 * rn(*p.shape)
        elif "conv1d" in name:
            v = 0.7 * rn(*p.shape)
        elif "norm" in name and last == "weight":
            v = 1.0 + 0.2 * rn(*p.shape)
        elif name == "model.embed_tokens.weight":
            v = rn(*p.shape)
        elif name == "model.per_layer_inputs.table":
            v = 0.3 * rn(*p.shape)
        elif p.ndim >= 2:
            gain = next((gv for k, gv in GAIN.items() if name.endswith(k)), 1.0)
            v = gain * rn(*p.shape) / math.sqrt(p.shape[-1])
        else:
            raise SystemExit(f"no rule for {name} {tuple(p.shape)}")
        assert v.shape == p.shape, name
        if last not in ("A_log", "dt_bias", "eta"):
            v = v.to(torch.bfloat16)
        tensors[name] = v.contiguous()

save_file(tensors, os.path.join(out, "model.safetensors"), metadata={"format": "pt"})
with open(os.path.join(out, "config.json"), "w") as f:
    json.dump(cfg, f, indent=2)
print("saved", out, len(tensors), "tensors")
