#!/usr/bin/env python3
# Builds a tiny random Ripple checkpoint in the layout of ricursiveAI/Ripple-65B-Preview
# (config.json with its keys, `model.safetensors` with its tensor names), with every feature of
# the model on: more heads than the width holds (8 heads of 16 on 64, 2 key / value heads),
# sliding layers with rotary positions and a window short enough to bind (8), full layers
# without positions (every fifth, as released), gated attention, value residuals, an embedding
# scale, an MLP clamp low enough to act (1.5), four norms per block, an untied head.
#
# usage: ripple_tiny.py OUT_DIR CODE_DIR [--plain]
#   CODE_DIR: `modeling_ricursive.py` and `configuration_ricursive.py` of
#   ricursiveAI/Ripple-65B-Preview. Matrices are stored in bf16, as released.
#   --plain: the same shapes without the optional features (no gate, no value residuals, no
#   embedding scale, no clamp, no window, a tied head), which the modeling code also allows.
# Then: ref_ripple.py OUT_DIR CODE_DIR ref.safetensors 96 16, and
#   cargo run --release --example lm_check -- OUT_DIR ref.safetensors [--chunk N]
import json, math, os, sys

import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ref_ripple import reference

out, code = sys.argv[1], sys.argv[2]
plain = "--plain" in sys.argv
os.makedirs(out, exist_ok=True)
n_layers = 10
cfg = {
    "architectures": ["RicursiveForCausalLM"], "model_type": "ricursive",
    "bos_token_id": 1, "eos_token_id": [2, 1], "pad_token_id": 0,
    "embed_scale": None if plain else 8.0, "ffn_clamp_threshold": None if plain else 1.5,
    "gate_full_precision": True, "head_dim": 16, "hidden_act": "silu", "hidden_size": 64,
    "intermediate_size": 256,
    "layer_types": ["full_attention" if i % 5 == 4 else "sliding_attention" for i in range(n_layers)],
    "max_position_embeddings": 2097152, "num_attention_heads": 8, "num_hidden_layers": n_layers,
    "num_key_value_heads": 2, "rms_norm_eps": 1e-06, "rope_theta": 10000.0,
    "sliding_window": None if plain else 8, "tie_word_embeddings": plain, "use_cache": True,
    "use_gated_attention": not plain, "use_valres": not plain, "vocab_size": 512,
}
M = reference(code)
C = sys.modules["ripple_ref.configuration_ricursive"]
skip = ("architectures", "model_type")
model = M.RicursiveForCausalLM(C.RicursiveConfig(**{k: v for k, v in cfg.items() if k not in skip}))
g = torch.Generator().manual_seed(65)


def rn(*shape):
    return torch.randn(tuple(shape), generator=g)


GAIN = {"o_proj.weight": 0.5, "down_proj.weight": 0.5, "lm_head.weight": 2.0}
tensors = {}
with torch.no_grad():
    for name, p in model.named_parameters():
        last = name.split(".")[-1]
        if name == "lm_head.weight" and plain:
            continue  # tied
        if last == "value_res_lambda":
            v = rn(*p.shape)
        elif "norm" in name and last == "weight":
            v = 1.0 + 0.2 * rn(*p.shape)
        elif name == "model.embed_tokens.weight":
            # Unit rows after the scale.
            v = rn(*p.shape) / (cfg["embed_scale"] or 1.0)
        elif p.ndim == 2:
            gain = next((gv for k, gv in GAIN.items() if name.endswith(k)), 1.0)
            v = gain * rn(*p.shape) / math.sqrt(p.shape[-1])
        else:
            raise SystemExit(f"no rule for {name} {tuple(p.shape)}")
        assert v.shape == p.shape, name
        tensors[name] = v.to(torch.bfloat16).contiguous()

save_file(tensors, os.path.join(out, "model.safetensors"), metadata={"format": "pt"})
with open(os.path.join(out, "config.json"), "w") as f:
    json.dump(cfg, f, indent=2)
print("saved", out, len(tensors), "tensors")
