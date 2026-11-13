#!/usr/bin/env python3
# Builds a tiny random ERNIE 5.5 checkpoint in the layout of baidu/ERNIE-5.5-Flash-Base-BF16
# (config.json with its keys, `model.safetensors` with its tensor names; --fp8: the matrices the
# FP8 release quantizes as F8_E4M3 with 128 x 128 block scales, `weight_scale_inv`), with every
# feature of the model: HCA layers (compressed keys beside a sliding window, sinks, grouped
# low-rank output with its gate, VHA), MLA layers whose DSA indexer has a top-k small enough that
# most queries are sparse, hyper-connections, one dense SiTU layer then latent MoE layers with two
# routers and per-expert scales, YaRN on the HCA layers, the MultiMax head, and the MTP layer.
#
# usage: ernie55_tiny.py OUT_DIR REAL_REPO_DIR [--f32] [--fp8] [--ratio R]
#   REAL_REPO_DIR holds tokenizer.json / tokenizer_config.json of baidu/ERNIE-5.5-Flash-Base.
#   Matrices are stored in bf16 (--f32: float32). --ratio: the HCA compression rate (default 8;
#   the release has 128).
# Then: ref_ernie55.py OUT_DIR MODEL_PY ref.safetensors 300 16, and
#   cargo run --release --example deepseek4_check -- OUT_DIR ref.safetensors [--chunk N]
import json, math, os, shutil, sys

import torch
from safetensors.torch import save_file

out, real = sys.argv[1], sys.argv[2]
f32, fp8 = "--f32" in sys.argv, "--fp8" in sys.argv
ratio = int(sys.argv[sys.argv.index("--ratio") + 1]) if "--ratio" in sys.argv else 8
os.makedirs(out, exist_ok=True)
for fn in ("tokenizer.json", "tokenizer_config.json", "generation_config.json"):
    if os.path.exists(os.path.join(real, fn)):
        shutil.copy(os.path.join(real, fn), os.path.join(out, fn))

ratios = [ratio, ratio, -2, ratio, ratio, -2, ratio, -2, -2]  # the last one: the MTP layer
cfg = {
    "architectures": ["Ernie5_5_MoeForCausalLM"], "model_type": "ernie5_5_moe",
    "torch_dtype": "float32" if f32 else "bfloat16", "bos_token_id": 1, "eos_token_id": 2, "pad_token_id": 0,
    "vocab_size": 201216, "hidden_size": 256, "num_hidden_layers": len(ratios) - 1, "num_nextn_predict_layers": 1,
    "num_attention_heads": 4, "num_key_value_heads": 1, "head_dim": 64, "v_head_dim": 64,
    "qk_rope_head_dim": 16, "qk_nope_head_dim": 48, "q_lora_rank": 48, "kv_lora_rank": 40,
    "o_groups": 2, "o_lora_rank": 32, "sliding_window": 16, "compress_ratios": ratios,
    "compress_rope_theta": 160000.0, "rope_theta": 8000000, "max_position_embeddings": 1048576,
    "rope_scaling": {"type": "yarn", "factor": 4, "beta_fast": 32, "beta_slow": 1,
                     "original_max_position_embeddings": 64},
    "hybrid_mla_attention": "mqa_dsa", "experimental_attention_variant": "dsv4_hybrid",
    "hybrid_mla_num_attention_heads": 4, "hybrid_mla_num_key_value_heads": 4, "hybrid_mla_q_lora_rank": 48,
    "hybrid_mla_kv_lora_rank": 40, "hybrid_mla_qk_nope_head_dim": 24, "hybrid_mla_qk_rope_head_dim": 16,
    "hybrid_mla_v_head_dim": 32, "index_n_heads": 4, "index_head_dim": 32, "index_topk": 24,
    "add_full_attention_sink_bias": True, "use_gated_attn": True, "gated_attn_act": "sigmoid",
    "gated_attn_use_q_lora": True, "use_vha_attention": True, "use_qk_norm": True, "qk_norm_type": "per_head",
    "enable_hyper_connections": True, "hc_mult": 4, "hc_sinkhorn_iters": 20, "hc_eps": 1e-06,
    # Large, so that the two RMS forms of the hyper-connections (1 / (sqrt(mean) + eps) before a
    # block, rsqrt(mean + eps) in the head) give different numbers.
    "rms_norm_eps": 0.01,
    "hidden_act": "situ", "activation_situ_beta": 1.0, "activation_situ_linear_beta": 2.0,
    "intermediate_size": 384, "first_k_dense_replace": 1, "moe_intermediate_size": 64, "moe_latent_size": 96,
    "latent_moe_use_norm": True, "moe_split_feature_routing": True, "n_routed_experts": 8, "n_shared_experts": 1,
    "num_experts_per_tok": 2, "norm_topk_prob": True, "scoring_func": "sigmoid", "topk_method": "noaux_tc",
    "n_group": 1, "topk_group": 1, "routed_scaling_factor": 3.0, "routed_scaling_factor_learnable": True,
    "multimax": "lm_head", "mtp_shared_last_layer": True, "tie_word_embeddings": False,
}
if fp8:
    cfg["quantization_config"] = {"activation_scheme": "dynamic", "fmt": "e4m3", "quant_method": "fp8",
                                  "scale_fmt": "ue8m0", "weight_block_size": [128, 128]}

c = cfg
H, V, hc = c["hidden_size"], c["vocab_size"], c["hc_mult"]
nh, hd, ql = c["num_attention_heads"], c["head_dim"], c["q_lora_rank"]
g_, orank = c["o_groups"], c["o_lora_rank"]
mh, mql, mr = c["hybrid_mla_num_attention_heads"], c["hybrid_mla_q_lora_rank"], c["hybrid_mla_kv_lora_rank"]
nope, rd, vd = c["hybrid_mla_qk_nope_head_dim"], c["hybrid_mla_qk_rope_head_dim"], c["hybrid_mla_v_head_dim"]
ni, di = c["index_n_heads"], c["index_head_dim"]
ne, mi, lat = c["n_routed_experts"], c["moe_intermediate_size"], c["moe_latent_size"]
mix = (2 + hc) * hc
gen = torch.Generator().manual_seed(42)
T = {}


def rn(*shape):
    return torch.randn(*shape, generator=gen)


def mat(name, n_out, n_in, gain=1.0, quant=True):
    """A matrix [n_out, n_in]; `quant`: one the FP8 release stores as F8_E4M3."""
    w = gain * rn(n_out, n_in) / math.sqrt(n_in)
    if fp8 and quant:
        # Block scales as the release: weight = fp8 * weight_scale_inv per 128 x 128 block.
        bo, bi = -(-n_out // 128), -(-n_in // 128)
        s = torch.zeros(bo, bi)
        q = torch.zeros(n_out, n_in)
        for i in range(bo):
            for j in range(bi):
                blk = w[i * 128:(i + 1) * 128, j * 128:(j + 1) * 128]
                s[i, j] = blk.abs().max() / 448.0
                q[i * 128:(i + 1) * 128, j * 128:(j + 1) * 128] = blk / s[i, j]
        T[name] = q.to(torch.float8_e4m3fn)
        T[name.removesuffix(".weight") + (".weight_scale_inv" if name.endswith(".weight") else "_scale_inv")] = s
    else:
        T[name] = w if f32 else w.to(torch.bfloat16)


def vec(name, n, kind="norm", f32_always=False):
    v = {"norm": lambda: 1.0 + 0.2 * rn(n), "small": lambda: 0.1 * rn(n), "unit": lambda: rn(n)}[kind]()
    T[name] = v if (f32 or f32_always) else v.to(torch.bfloat16)


def hyper(p):
    T[p + "mapping_proj.weight"] = (rn(mix, hc * H) / math.sqrt(hc * H) * 4.0).t().contiguous()  # [in, out]
    T[p + "bias"] = 0.5 * rn(mix)
    for a in ("pre", "post", "res"):
        T[p + f"alpha_{a}"] = 0.5 + torch.rand(1, generator=gen)


def hc_head(p):
    T[p + "hc_head_fn"] = (rn(hc, hc * H) / math.sqrt(hc * H) * 4.0).t().contiguous()
    T[p + "hc_head_base"] = 0.5 * rn(hc)
    T[p + "hc_head_scale"] = 0.5 + torch.rand(1, generator=gen)


T["model.embed_tokens.weight"] = rn(V, H) if f32 else rn(V, H).to(torch.bfloat16)
mat("lm_head.weight", V, H, 4.0, quant=False)
vec("model.norm.weight", H)
hc_head("model.mhc_contract.")
# Inside the range of the logits, so that all four pieces of the MultiMax act (and small enough
# that it stays increasing over that range).
T["model.lm_head.multimax_ranges"] = torch.tensor([-1.5, 1.5, -3.0, 3.0]).to(torch.float32 if f32 else torch.bfloat16)
T["model.lm_head.multimax_ts"] = torch.tensor([0.3, -0.2, 0.01, -0.01]).to(torch.float32 if f32 else torch.bfloat16)

for l, r in enumerate(ratios):
    p = f"model.layers.{l}."
    a = p + "self_attn."
    mtp = l >= c["num_hidden_layers"]
    hp = p + ("transformer_layer." if mtp else "")
    hyper(hp + "self_attention_hyper_connection.")
    hyper(hp + "mlp_hyper_connection.")
    vec(p + "input_layernorm.weight", H)
    vec(p + "post_attention_layernorm.weight", H)
    T[a + "vha_postmix_U"] = (0.3 * rn(nh, 16)).to(torch.float32 if f32 else torch.bfloat16)
    T[a + "vha_postmix_V"] = (0.3 * rn(nh, 16)).to(torch.float32 if f32 else torch.bfloat16)
    if r > 0:
        mat(a + "linear_q_down_proj.weight", ql, H, 2.0)
        vec(a + "q_layernorm.weight", ql)
        mat(a + "linear_q_up_proj.weight", nh * hd, ql, 2.0)
        mat(a + "linear_kv_proj.weight", hd, H, 2.0)
        vec(a + "kv_layernorm.weight", hd)
        mat(a + "linear_o_group_proj", g_ * orank, nh * hd // g_)
        mat(a + "o_proj.weight", H, g_ * orank)
        mat(a + "gate_proj.weight", g_ * orank, ql, 3.0)
        T[a + "core_attention.attn_sink"] = rn(nh)
        cp = a + "core_attention.compressor."
        T[cp + "ape"] = 0.1 * rn(r, hd)
        mat(cp + "linear_wkv.weight", hd, H, 2.0, quant=False)
        mat(cp + "linear_wgate.weight", hd, H, 3.0, quant=False)
        vec(cp + "norm.weight", hd)
    else:
        mat(a + "q_a_proj.weight", mql, H, 2.0)
        vec(a + "q_a_layernorm.weight", mql)
        mat(a + "q_b_proj.weight", mh * (nope + rd), mql, 2.0)
        mat(a + "kv_a_proj_with_mqa.weight", mr + rd, H, 2.0)
        vec(a + "kv_a_layernorm.weight", mr)
        mat(a + "kv_b_proj.weight", mh * (nope + vd), mr, 2.0)
        mat(a + "o_proj.weight", H, mh * vd)
        mat(a + "gate_proj.weight", mh * vd, mql, 3.0)
        vec(a + "core_attention.softmax_offset", mh, "unit")
        ip = a + "core_attention.indexer."
        mat(ip + "wq_b.weight", ni * di, mql, 2.0)
        mat(ip + "wk.weight", di, H, 2.0, quant=False)
        vec(ip + "k_norm.weight", di)
        vec(ip + "k_norm.bias", di, "small")
        mat(ip + "weights_proj.weight", ni, H, 2.0, quant=False)
    if l < c["first_k_dense_replace"]:
        inter = c["intermediate_size"]
        mat(p + "mlp.gate_proj.weight", inter, H, 3.0)
        mat(p + "mlp.up_proj.weight", inter, H, 3.0)
        mat(p + "mlp.down_proj.weight", H, inter)
    else:
        m = p + "block_sparse_moe."
        mat(m + "gate.weight", ne, H, 3.0, quant=False)
        mat(m + "gate.weight_1", ne, H, 3.0, quant=False)
        T[m + "e_score_correction_bias"] = 0.3 * rn(ne)
        T[m + "gate.routed_scaling_factor_param"] = 3.0 + 0.5 * rn(ne)
        mat(m + "fc1_latent_proj.weight", lat, H, 2.0)
        mat(m + "fc2_latent_proj.weight", H, lat)
        vec(m + "latent_norm.weight", lat)
        for e in range(ne):
            mat(m + f"experts.{e}.w1.weight", mi, lat, 3.0)
            mat(m + f"experts.{e}.w3.weight", mi, lat, 3.0)
            mat(m + f"experts.{e}.w2.weight", lat, mi)
        mat(m + "shared_experts.w1.weight", mi, H, 3.0)
        mat(m + "shared_experts.w3.weight", mi, H, 3.0)
        mat(m + "shared_experts.w2.weight", H, mi)
    if mtp:
        mat(p + "e_proj.weight", H, H)
        mat(p + "h_proj.weight", H, H)
        vec(p + "enorm.weight", H)
        vec(p + "hnorm.weight", H)
        vec(p + "shared_head.norm.weight", H)
        hc_head(p)

save_file({k: v.contiguous() for k, v in T.items()}, os.path.join(out, "model.safetensors"), metadata={"format": "pt"})
with open(os.path.join(out, "config.json"), "w") as f:
    json.dump(cfg, f, indent=2)
print("saved", out, len(T), "tensors")
