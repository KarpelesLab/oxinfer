#!/usr/bin/env python3
# Builds a tiny random DeepSeek V4 checkpoint in the official layout (config.json with DeepSeek's
# keys, `model.safetensors` with the names of deepseek-ai/DeepSeek-V4-Flash: `layers.N.attn.wq_a.weight`,
# `layers.N.ffn.experts.E.w1.weight`, ...), with every architectural feature of the real model:
# hash-routed first layers, sliding-window-only layers, CSA layers (overlapping 4-token compression
# + lightning indexer, top-k small enough that most queries are sparse), HCA layers (a short
# compression rate so a 300-token prompt has many entries), hyper-connections, attention sinks,
# grouped output projection, YaRN on the compressed layers, clamped SwiGLU experts.
# transformers loads it back through its conversion mapping (checked here, tensor by tensor).
#
# usage: deepseek4_tiny.py OUT_DIR REAL_REPO_DIR [--hca N] [--bf16] [--dspark]
#   REAL_REPO_DIR holds tokenizer.json / tokenizer_config.json of deepseek-ai/DeepSeek-V4-Flash.
#   --dspark: also DSpark draft stages (`mtp.0..2`, as deepseek-ai/DeepSeek-V4-Flash-DSpark), for
#   tools/ref_dspark_ds4.py (the official DSpark inference/model.py) and tools/dspark_ds4_gguf.py.
# Then: ref_deepseek4.py OUT_DIR MODEL_PY ref.safetensors 300 16, and
#   cargo run --release --example deepseek4_check -- OUT_DIR ref.safetensors [--chunk N]
import json, math, os, shutil, sys

import torch
from safetensors.torch import save_file

from transformers import DeepseekV4Config, DeepseekV4ForCausalLM

out, real = sys.argv[1], sys.argv[2]
hca = int(sys.argv[sys.argv.index("--hca") + 1]) if "--hca" in sys.argv else 16
bf16 = "--bf16" in sys.argv
dspark = "--dspark" in sys.argv
os.makedirs(out, exist_ok=True)
torch.manual_seed(1234)

ratios = [0, 4, hca, 4, hca, 0]
n = len(ratios)
kind = {0: "sliding_attention", 4: "compressed_sparse_attention", hca: "heavily_compressed_attention"}
official = {
    "architectures": ["DeepseekV4ForCausalLM"], "model_type": "deepseek_v4",
    "vocab_size": 129280, "hidden_size": 256, "num_hidden_layers": n, "num_attention_heads": 4,
    "num_key_value_heads": 1, "head_dim": 128, "qk_rope_head_dim": 32, "q_lora_rank": 64,
    "o_groups": 2, "o_lora_rank": 48, "moe_intermediate_size": 64, "n_routed_experts": 8,
    "n_shared_experts": 1, "num_experts_per_tok": 2, "num_hash_layers": 1, "scoring_func": "sqrtsoftplus",
    "norm_topk_prob": True, "routed_scaling_factor": 1.5, "swiglu_limit": 1.5, "hidden_act": "silu",
    "sliding_window": 16, "compress_ratios": ratios + [0], "compress_rope_theta": 160000,
    "rope_theta": 10000,
    "rope_scaling": {"type": "yarn", "factor": 4, "original_max_position_embeddings": 64, "beta_fast": 32,
                     "beta_slow": 1},
    "index_n_heads": 16, "index_head_dim": 64, "index_topk": 8, "hc_mult": 4, "hc_sinkhorn_iters": 20,
    "hc_eps": 1e-6, "rms_norm_eps": 1e-6, "num_nextn_predict_layers": 0, "tie_word_embeddings": False,
    "bos_token_id": 0, "eos_token_id": 1, "max_position_embeddings": 1048576,
    "torch_dtype": "bfloat16" if bf16 else "float32",
}
MARKOV = 32
if dspark:
    official.update({
        "compress_ratios": ratios + [0, 0, 0], "num_nextn_predict_layers": 3, "dspark_block_size": 5,
        "dspark_noise_token_id": 128799, "dspark_target_layer_ids": [2, 3, 5], "dspark_markov_rank": MARKOV,
    })
# transformers' own keys (its layer types only know the rates 4 and 128 by default).
extra = {
    "layer_types": [kind[r] for r in ratios],
    "compress_rates": {"compressed_sparse_attention": 4, "heavily_compressed_attention": hca},
    "mlp_layer_types": ["hash_moe"] + ["moe"] * (n - 1),
}


def make(over):
    d = json.loads(json.dumps(dict(official, **extra)))  # a copy: the config edits nested dicts
    d.update(over)
    d.pop("compress_ratios")
    return DeepseekV4Config(**{k: v for k, v in d.items()
                               if k not in ("architectures", "model_type", "torch_dtype") and not k.startswith("dspark")})


cfg = make({})
model = DeepseekV4ForCausalLM(cfg).float().eval()

g = torch.Generator().manual_seed(42)


def rn(*shape):
    return torch.randn(*shape, generator=g)


def init(model, cfg):
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "norm" in name and name.endswith("weight"):
                p.copy_(1.0 + 0.2 * rn(*p.shape))
            elif name.endswith("sinks"):
                p.copy_(rn(*p.shape))
            elif name.endswith("position_bias"):
                p.copy_(0.5 * rn(*p.shape))
            elif name.endswith(".base") or name.endswith("hc_base"):
                p.copy_(0.5 * rn(*p.shape))
            elif name.endswith(".scale") or name.endswith("hc_scale"):
                p.copy_(0.5 + torch.rand(p.shape, generator=g))
            elif name.endswith(".fn") or name.endswith("hc_fn"):
                p.copy_(rn(*p.shape) / math.sqrt(p.shape[-1]) * 4.0)
            elif "embed_tokens" in name:
                p.copy_(rn(*p.shape))
            elif p.ndim >= 2:
                fan_in = p.shape[-1]
                gain = 1.0
                if "q_b_proj" in name or "kv_proj" in name or "q_a_proj" in name:
                    gain = 2.0
                if "mlp.gate.weight" in name:
                    gain = 3.0
                if "lm_head" in name:
                    gain = 4.0
                if "gate_proj" in name and "compressor" in name:
                    gain = 3.0
                p.copy_(gain * rn(*p.shape) / math.sqrt(fan_in))
            else:
                p.copy_(0.1 * rn(*p.shape))
        for name, b in model.named_buffers():
            if name.endswith("e_score_correction_bias"):
                b.copy_(0.3 * rn(*b.shape))
            elif name.endswith("tid2eid"):
                # Distinct experts per token, as the real table.
                v = torch.stack([torch.randperm(cfg.n_routed_experts, generator=g)[: cfg.num_experts_per_tok]
                                 for _ in range(b.shape[0])])
                b.copy_(v)


init(model, cfg)

# The official names of transformers' parameters.
REN = [
    ("model.embed_tokens.weight", "embed.weight"), ("lm_head.weight", "head.weight"),
    ("model.norm.weight", "norm.weight"), ("model.hc_head.hc_fn", "hc_head_fn"),
    ("model.hc_head.hc_base", "hc_head_base"), ("model.hc_head.hc_scale", "hc_head_scale"),
]
LAYER = [
    ("input_layernorm.weight", "attn_norm.weight"), ("post_attention_layernorm.weight", "ffn_norm.weight"),
    ("attn_hc.fn", "hc_attn_fn"), ("attn_hc.base", "hc_attn_base"), ("attn_hc.scale", "hc_attn_scale"),
    ("ffn_hc.fn", "hc_ffn_fn"), ("ffn_hc.base", "hc_ffn_base"), ("ffn_hc.scale", "hc_ffn_scale"),
    ("self_attn.sinks", "attn.attn_sink"),
    ("self_attn.q_a_proj.weight", "attn.wq_a.weight"), ("self_attn.q_a_norm.weight", "attn.q_norm.weight"),
    ("self_attn.q_b_proj.weight", "attn.wq_b.weight"), ("self_attn.kv_proj.weight", "attn.wkv.weight"),
    ("self_attn.kv_norm.weight", "attn.kv_norm.weight"), ("self_attn.o_a_proj.weight", "attn.wo_a.weight"),
    ("self_attn.o_b_proj.weight", "attn.wo_b.weight"),
    ("self_attn.compressor.kv_proj.weight", "attn.compressor.wkv.weight"),
    ("self_attn.compressor.gate_proj.weight", "attn.compressor.wgate.weight"),
    ("self_attn.compressor.position_bias", "attn.compressor.ape"),
    ("self_attn.compressor.kv_norm.weight", "attn.compressor.norm.weight"),
    ("self_attn.compressor.indexer.kv_proj.weight", "attn.indexer.compressor.wkv.weight"),
    ("self_attn.compressor.indexer.gate_proj.weight", "attn.indexer.compressor.wgate.weight"),
    ("self_attn.compressor.indexer.position_bias", "attn.indexer.compressor.ape"),
    ("self_attn.compressor.indexer.kv_norm.weight", "attn.indexer.compressor.norm.weight"),
    ("self_attn.compressor.indexer.q_b_proj.weight", "attn.indexer.wq_b.weight"),
    ("self_attn.compressor.indexer.scorer.weights_proj.weight", "attn.indexer.weights_proj.weight"),
    ("mlp.gate.weight", "ffn.gate.weight"), ("mlp.gate.e_score_correction_bias", "ffn.gate.bias"),
    ("mlp.gate.tid2eid", "ffn.gate.tid2eid"),
    ("mlp.shared_experts.gate_proj.weight", "ffn.shared_experts.w1.weight"),
    ("mlp.shared_experts.up_proj.weight", "ffn.shared_experts.w3.weight"),
    ("mlp.shared_experts.down_proj.weight", "ffn.shared_experts.w2.weight"),
]
dt = torch.bfloat16 if bf16 else torch.float32


def keep(name, v):
    if name.endswith("tid2eid"):
        return v.to(torch.int32).contiguous()
    if v.ndim >= 2 and not any(s in name for s in ("hc_", "ape")):
        v = v.to(dt)  # the reference keeps the hc / ape / norm / sink / bias tensors in float32
    return v.contiguous()


def official_names(sd, layer_prefix, tensors):
    """transformers' state dict `sd` into `tensors` under DeepSeek's names (layers as `layer_prefix`N)."""
    for k, v in sd.items():
        if k.endswith("inv_freq") or "rotary_emb" in k:
            continue
        name = dict(REN).get(k)
        if name is None:
            l, rest = k.removeprefix("model.layers.").split(".", 1)
            p = f"{layer_prefix}{l}."
            if rest == "mlp.experts.gate_up_proj":
                mi = cfg.moe_intermediate_size
                for e in range(v.shape[0]):
                    tensors[f"{p}ffn.experts.{e}.w1.weight"] = v[e, :mi].to(dt).contiguous()
                    tensors[f"{p}ffn.experts.{e}.w3.weight"] = v[e, mi:].to(dt).contiguous()
                continue
            if rest == "mlp.experts.down_proj":
                for e in range(v.shape[0]):
                    tensors[f"{p}ffn.experts.{e}.w2.weight"] = v[e].to(dt).contiguous()
                continue
            name = p + dict(LAYER)[rest]
        tensors[name] = keep(name, v)


sd = model.state_dict()
tensors = {}
official_names(sd, "layers.", tensors)

if dspark:
    # Three window-only stages with their own hc head and norm (the last), the feature projection
    # (the first), and the Markov / confidence heads (the last).
    dcfg = make({"num_hidden_layers": 3, "layer_types": ["sliding_attention"] * 3, "mlp_layer_types": ["moe"] * 3})
    dm = DeepseekV4ForCausalLM(dcfg).float().eval()
    init(dm, dcfg)
    dsd = {k: v for k, v in dm.state_dict().items() if k.startswith("model.layers.")}
    official_names(dsd, "mtp.", tensors)
    h, v = official["hidden_size"], official["vocab_size"]
    nt = len(official["dspark_target_layer_ids"])
    hc = official["hc_mult"]
    extra_t = {
        "mtp.0.main_proj.weight": rn(h, nt * h) / math.sqrt(nt * h) * 2.0,
        "mtp.0.main_norm.weight": 1.0 + 0.2 * rn(h),
        "mtp.2.norm.weight": 1.0 + 0.2 * rn(h),
        "mtp.2.hc_head_fn": rn(hc, hc * h) / math.sqrt(hc * h) * 4.0,
        "mtp.2.hc_head_base": 0.5 * rn(hc),
        "mtp.2.hc_head_scale": 0.5 + torch.rand(1, generator=g),
        "mtp.2.markov_head.markov_w1.weight": rn(v, MARKOV),
        "mtp.2.markov_head.markov_w2.weight": rn(v, MARKOV) / math.sqrt(MARKOV) * 2.0,
        "mtp.2.confidence_head.proj.weight": rn(1, h + MARKOV) / math.sqrt(h + MARKOV),
    }
    for k, t in extra_t.items():
        tensors[k] = keep(k, t)

save_file(tensors, os.path.join(out, "model.safetensors"), metadata={"format": "pt"})
with open(os.path.join(out, "config.json"), "w") as f:
    json.dump(dict(official, **extra), f, indent=2)
for fn in ("tokenizer.json", "tokenizer_config.json"):
    shutil.copy(os.path.join(real, fn), os.path.join(out, fn))

# transformers reads the official layout back (its conversion mapping) to the same values.
back = DeepseekV4ForCausalLM.from_pretrained(out, dtype=torch.float32).eval()
bsd = back.state_dict()
worst = 0.0
for k, v in sd.items():
    if k.endswith("inv_freq"):
        continue
    if v.dtype.is_floating_point:
        # Stored in `dt`, or kept in float32 (hc / ape / norms / sinks / biases).
        b = bsd[k].float()
        d = torch.minimum((b - v.to(dt).float()).abs(), (b - v.float()).abs()).max().item()
    else:
        d = (bsd[k].float() - v.float()).abs().max().item()
    worst = max(worst, d)
    assert d == 0.0, (k, d)
print("saved", out, len(tensors), "tensors; reload max diff", worst)
