"""Writes a tiny GLM-5.3-Flash / GLM-5.3 checkpoint (tools/glm5_tiny.py) as a llama.cpp GGUF file,
with the metadata keys and tensor names of the published `glm5-next` / `glm-dsa` files
(ggml-org/GLM-5.3-Flash-GGUF, unsloth/GLM-5.3-GGUF): `ssm_a` holds `-exp(A_log)`, `kv_b_proj`
is split per head into `attn_k_b` (the key rows transposed) and `attn_v_b`, the experts are
stacked. float32 tensors, no tokenizer: for `examples/glm5_check.rs`, which must print the same
numbers for the GGUF as for the folder.

usage: glm5_gguf.py MODEL_DIR OUT.gguf      (needs the `gguf` package)
"""
import json, os, re, sys
import numpy as np
import gguf
from safetensors.numpy import load_file

src, out = sys.argv[1], sys.argv[2]
cfg = json.load(open(os.path.join(src, "config.json")))
t = cfg.get("text_config", cfg)
flash = t["model_type"].startswith("glm5_next")
arch = "glm5-next" if flash else "glm-dsa"
sd = load_file(os.path.join(src, "model.safetensors"))
P = "model.language_model." if flash else "model."
n = t["num_hidden_layers"]
nh, lat, dn, dr, vd = t["num_attention_heads"], t["kv_lora_rank"], t["qk_nope_head_dim"], t["qk_rope_head_dim"], t["v_head_dim"]
kda = [x == "linear_attention" for x in t.get("layer_types", ["x"] * n)]

w = gguf.GGUFWriter(out, arch)
k = lambda s: f"{arch}.{s}"
w.add_uint32(k("block_count"), n)
w.add_uint32(k("context_length"), t["max_position_embeddings"])
w.add_uint32(k("embedding_length"), t["hidden_size"])
w.add_uint32(k("feed_forward_length"), t["intermediate_size"])
w.add_uint32(k("attention.head_count"), nh)
w.add_float32(k("attention.layer_norm_rms_epsilon"), t["rms_norm_eps"])
w.add_uint32(k("expert_count"), t["n_routed_experts"])
w.add_uint32(k("expert_used_count"), t["num_experts_per_tok"])
w.add_uint32(k("expert_group_count"), 1)
w.add_uint32(k("expert_group_used_count"), 1)
w.add_uint32(k("expert_gating_func"), 2)
w.add_uint32(k("vocab_size"), t["vocab_size"])
w.add_uint32(k("attention.q_lora_rank"), t["q_lora_rank"])
w.add_uint32(k("attention.kv_lora_rank"), lat)
w.add_uint32(k("rope.dimension_count"), dr)
w.add_uint32(k("attention.key_length"), lat + dr)
w.add_uint32(k("attention.value_length"), lat)
w.add_uint32(k("attention.key_length_mla"), dn + dr)
w.add_uint32(k("attention.value_length_mla"), vd)
w.add_uint32(k("attention.indexer.head_count"), t["index_n_heads"])
w.add_uint32(k("attention.indexer.key_length"), t["index_head_dim"])
w.add_uint32(k("attention.indexer.top_k"), t["index_topk"])
w.add_uint32(k("leading_dense_block_count"), t["mlp_layer_types"].index("sparse"))
w.add_uint32(k("expert_feed_forward_length"), t["moe_intermediate_size"])
w.add_uint32(k("expert_shared_count"), t["n_shared_experts"])
w.add_float32(k("expert_weights_scale"), t["routed_scaling_factor"])
w.add_bool(k("expert_weights_norm"), True)
if flash:
    w.add_array(k("attention.head_count_kv"), [0 if x else 1 for x in kda])
    w.add_float32(k("attention.layer_norm_epsilon"), 1e-6)
    w.add_uint32(k("ssm.conv_kernel"), t["linear_conv_kernel_dim"])
    w.add_uint32(k("kda.head_dim"), t["linear_head_dim"])
    w.add_float32(k("kda.gate_lower_bound"), t["linear_lower_bound"])
    w.add_uint32(k("attention.indexer.kpool"), t["index_kpool"])
    w.add_bool(k("attention.indexer.kpool_select_tail"), t["index_kpool_always_select_tail"])
    w.add_array(k("attention.indexer.types"), [x == "full" for x in t["indexer_types"]])
    w.add_uint32(k("hyper_connection.count"), t["hc_mult"])
    w.add_uint32(k("hyper_connection.sinkhorn_iterations"), t["hc_sinkhorn_iters"])
    w.add_float32(k("hyper_connection.epsilon"), t["hc_eps"])
    w.add_array(k("swiglu_clamp_exp"), [float(t["swiglu_limit"])] * n)
    w.add_array(k("swiglu_clamp_shexp"), [float(t["swiglu_limit"])] * n)
else:
    w.add_float32(k("rope.freq_base"), t["rope_parameters"]["rope_theta"])
    w.add_uint32(k("nextn_predict_layers"), 0)


def put(name, a):
    w.add_tensor(name, np.ascontiguousarray(a, dtype=np.float32))


put("token_embd.weight", sd[P + "embed_tokens.weight"])
put("output.weight", sd["lm_head.weight"])
put("output_norm.weight", sd[P + "norm.weight"])
A = {
    "input_layernorm.weight": "attn_norm.weight", "post_attention_layernorm.weight": "ffn_norm.weight",
    "self_attn.q_proj.weight": "attn_q.weight", "self_attn.k_proj.weight": "attn_k.weight",
    "self_attn.v_proj.weight": "attn_v.weight", "self_attn.o_proj.weight": "attn_output.weight",
    "self_attn.dt_bias": "ssm_dt.bias", "self_attn.b_proj.weight": "ssm_beta.weight",
    "self_attn.f_a_proj.weight": "ssm_f_a.weight", "self_attn.f_b_proj.weight": "ssm_f_b.weight",
    "self_attn.g_a_proj.weight": "ssm_g_a.weight", "self_attn.g_b_proj.weight": "ssm_g_b.weight",
    "self_attn.o_norm.weight": "ssm_norm.weight", "self_attn.q_conv1d.weight": "ssm_conv1d_q.weight",
    "self_attn.k_conv1d.weight": "ssm_conv1d_k.weight", "self_attn.v_conv1d.weight": "ssm_conv1d_v.weight",
    "self_attn.q_a_proj.weight": "attn_q_a.weight", "self_attn.q_a_layernorm.weight": "attn_q_a_norm.weight",
    "self_attn.q_b_proj.weight": "attn_q_b.weight", "self_attn.kv_a_proj_with_mqa.weight": "attn_kv_a_mqa.weight",
    "self_attn.kv_a_layernorm.weight": "attn_kv_a_norm.weight",
    "self_attn.indexer.wk.weight": "indexer.attn_k.weight", "self_attn.indexer.wq_b.weight": "indexer.attn_q_b.weight",
    "self_attn.indexer.k_norm.weight": "indexer.k_norm.weight", "self_attn.indexer.k_norm.bias": "indexer.k_norm.bias",
    "self_attn.indexer.weights_proj.weight": "indexer.proj.weight",
    "self_attn.indexer.index_kpool_compress_ape": "indexer_compressor_ape.weight",
    "self_attn.indexer.index_kpool_compress_gate": "indexer_compressor_gate.weight",
    "mlp.gate_proj.weight": "ffn_gate.weight", "mlp.up_proj.weight": "ffn_up.weight", "mlp.down_proj.weight": "ffn_down.weight",
    "mlp.gate.weight": "ffn_gate_inp.weight", "mlp.gate.e_score_correction_bias": "exp_probs_b.bias",
    "mlp.shared_experts.gate_proj.weight": "ffn_gate_shexp.weight", "mlp.shared_experts.up_proj.weight": "ffn_up_shexp.weight",
    "mlp.shared_experts.down_proj.weight": "ffn_down_shexp.weight",
}
for hc in ["hc_attn_fn", "hc_attn_base", "hc_attn_scale", "hc_ffn_fn", "hc_ffn_base", "hc_ffn_scale"]:
    A[hc] = hc + ".weight"
for l in range(n):
    lp = f"{P}layers.{l}."
    names = [x[len(lp):] for x in sd if x.startswith(lp)]
    for nm in names:
        a = sd[lp + nm]
        if nm in A:
            put(f"blk.{l}.{A[nm]}", a)
        elif nm == "self_attn.A_log":
            put(f"blk.{l}.ssm_a", -np.exp(a))
        elif nm == "self_attn.kv_b_proj.weight":
            kv = a.reshape(nh, dn + vd, lat)
            put(f"blk.{l}.attn_k_b.weight", kv[:, :dn].transpose(0, 2, 1))  # [heads, latent, nope]
            put(f"blk.{l}.attn_v_b.weight", kv[:, dn:])  # [heads, value, latent]
        elif not re.match(r"mlp\.experts\.", nm):
            raise SystemExit(f"unmapped tensor {lp}{nm}")
    if lp + "mlp.experts.0.gate_proj.weight" in sd:
        for part, g in [("gate_proj", "ffn_gate_exps"), ("up_proj", "ffn_up_exps"), ("down_proj", "ffn_down_exps")]:
            put(f"blk.{l}.{g}.weight", np.stack([sd[f"{lp}mlp.experts.{e}.{part}.weight"] for e in range(t["n_routed_experts"])]))
    elif lp + "mlp.experts.gate_up_proj" in sd:
        gu = sd[lp + "mlp.experts.gate_up_proj"]
        mi = gu.shape[1] // 2
        put(f"blk.{l}.ffn_gate_exps.weight", gu[:, :mi])
        put(f"blk.{l}.ffn_up_exps.weight", gu[:, mi:])
        put(f"blk.{l}.ffn_down_exps.weight", sd[lp + "mlp.experts.down_proj"])
w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
print("wrote", out)
