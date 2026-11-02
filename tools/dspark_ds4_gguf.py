#!/usr/bin/env python3
# Writes the DSpark draft stages of a (tiny, float) DeepSeek V4 checkpoint in DeepSeek's layout
# (`mtp.N.*`, as tools/deepseek4_tiny.py --dspark makes them) as a llama.cpp `dflash` GGUF with the
# names and metadata of llama.cpp's converter (`convert_hf_to_gguf.py --dspark`), all in float32.
# The real draft (FP8 / FP4) goes through llama.cpp's converter instead.
#
# usage: dspark_ds4_gguf.py MODEL_DIR OUT.gguf GGUF_PY_DIR
import json, sys

import numpy as np
from safetensors.torch import load_file

d, out = sys.argv[1], sys.argv[2]
sys.path.insert(0, sys.argv[3])
import gguf  # noqa: E402

cfg = json.load(open(f"{d}/config.json"))
st = {k: v.float().numpy() for k, v in load_file(f"{d}/model.safetensors").items() if k.startswith("mtp.")}
ns = cfg["num_nextn_predict_layers"]
rs = cfg.get("rope_scaling") or {}
w = gguf.GGUFWriter(out, "dflash")
A = "dflash"
w.add_uint32(f"{A}.block_count", ns)
w.add_uint32(f"{A}.embedding_length", cfg["hidden_size"])
w.add_uint32(f"{A}.attention.head_count", cfg["num_attention_heads"])
w.add_uint32(f"{A}.attention.head_count_kv", 1)
w.add_uint32(f"{A}.attention.key_length", cfg["head_dim"])
w.add_uint32(f"{A}.attention.value_length", cfg["head_dim"])
w.add_uint32(f"{A}.rope.dimension_count", cfg["qk_rope_head_dim"])
w.add_float32(f"{A}.rope.freq_base", cfg["rope_theta"])
w.add_float32(f"{A}.attention.layer_norm_rms_epsilon", cfg["rms_norm_eps"])
w.add_uint32(f"{A}.expert_count", cfg["n_routed_experts"])
w.add_uint32(f"{A}.expert_used_count", cfg["num_experts_per_tok"])
w.add_uint32(f"{A}.expert_gating_func", 4)
w.add_uint32(f"{A}.attention.q_lora_rank", cfg["q_lora_rank"])
w.add_uint32(f"{A}.attention.sliding_window", cfg["sliding_window"])
w.add_uint32(f"{A}.expert_feed_forward_length", cfg["moe_intermediate_size"])
w.add_uint32(f"{A}.expert_shared_count", cfg["n_shared_experts"])
w.add_float32(f"{A}.expert_weights_scale", cfg["routed_scaling_factor"])
w.add_bool(f"{A}.expert_weights_norm", cfg["norm_topk_prob"])
w.add_array(f"{A}.swiglu_clamp_exp", [float(cfg["swiglu_limit"])] * ns)
w.add_array(f"{A}.swiglu_clamp_shexp", [float(cfg["swiglu_limit"])] * ns)
w.add_uint32(f"{A}.attention.indexer.head_count", cfg["index_n_heads"])
w.add_uint32(f"{A}.attention.indexer.key_length", cfg["index_head_dim"])
w.add_uint32(f"{A}.attention.indexer.top_k", cfg["index_topk"])
w.add_uint32(f"{A}.attention.output_group_count", cfg["o_groups"])
w.add_uint32(f"{A}.attention.output_lora_rank", cfg["o_lora_rank"])
w.add_array(f"{A}.attention.compress_ratios", [0] * ns)
w.add_float32(f"{A}.attention.compress_rope_freq_base", cfg["compress_rope_theta"])
w.add_uint32(f"{A}.hyper_connection.count", cfg["hc_mult"])
w.add_uint32(f"{A}.hyper_connection.sinkhorn_iterations", cfg["hc_sinkhorn_iters"])
w.add_float32(f"{A}.hyper_connection.epsilon", cfg["hc_eps"])
w.add_uint32(f"{A}.hash_layer_count", 0)
w.add_uint32(f"{A}.block_size", cfg["dspark_block_size"])
w.add_array(f"{A}.target_layers", [l + 1 for l in cfg["dspark_target_layer_ids"]])
w.add_uint32(f"{A}.vocab_size", cfg["vocab_size"])
w.add_uint32("tokenizer.ggml.mask_token_id", cfg["dspark_noise_token_id"])
if rs:
    w.add_string(f"{A}.rope.scaling.type", "yarn")
    w.add_float32(f"{A}.rope.scaling.factor", rs["factor"])
    w.add_uint32(f"{A}.rope.scaling.original_context_length", rs["original_max_position_embeddings"])

ROOT = {
    "mtp.0.main_proj.weight": "fc.weight", "mtp.0.main_norm.weight": "enc.output_norm.weight",
    f"mtp.{ns - 1}.norm.weight": "output_norm.weight", f"mtp.{ns - 1}.hc_head_fn": "output_hc_fn.weight",
    f"mtp.{ns - 1}.hc_head_base": "output_hc_base.weight", f"mtp.{ns - 1}.hc_head_scale": "output_hc_scale.weight",
    f"mtp.{ns - 1}.markov_head.markov_w1.weight": "markov_w1.weight",
    f"mtp.{ns - 1}.markov_head.markov_w2.weight": "markov_w2.weight",
    f"mtp.{ns - 1}.confidence_head.proj.weight": "conf_proj.weight",
}
LAYER = {
    "attn_norm.weight": "attn_norm.weight", "ffn_norm.weight": "ffn_norm.weight",
    "hc_attn_fn": "hc_attn_fn.weight", "hc_attn_base": "hc_attn_base.weight", "hc_attn_scale": "hc_attn_scale.weight",
    "hc_ffn_fn": "hc_ffn_fn.weight", "hc_ffn_base": "hc_ffn_base.weight", "hc_ffn_scale": "hc_ffn_scale.weight",
    "attn.attn_sink": "attn_sinks.weight", "attn.wq_a.weight": "attn_q_a.weight",
    "attn.q_norm.weight": "attn_q_a_norm.weight", "attn.wq_b.weight": "attn_q_b.weight",
    "attn.wkv.weight": "attn_kv.weight", "attn.kv_norm.weight": "attn_kv_a_norm.weight",
    "attn.wo_b.weight": "attn_output_b.weight", "ffn.gate.weight": "ffn_gate_inp.weight",
    "ffn.gate.bias": "exp_probs_b.bias", "ffn.shared_experts.w1.weight": "ffn_gate_shexp.weight",
    "ffn.shared_experts.w2.weight": "ffn_down_shexp.weight", "ffn.shared_experts.w3.weight": "ffn_up_shexp.weight",
}
f32 = lambda a: np.ascontiguousarray(a, dtype=np.float32)  # noqa: E731
for k, name in ROOT.items():
    w.add_tensor(name, f32(st[k]))
G, R = cfg["o_groups"], cfg["o_lora_rank"]
for l in range(ns):
    p = f"mtp.{l}."
    for k, name in LAYER.items():
        w.add_tensor(f"blk.{l}.{name}", f32(st[p + k]))
    wo_a = f32(st[p + "attn.wo_a.weight"])
    w.add_tensor(f"blk.{l}.attn_output_a.weight", wo_a.reshape(G, R, wo_a.shape[-1]))
    ne = cfg["n_routed_experts"]
    for proj, name in (("w1", "ffn_gate_exps"), ("w2", "ffn_down_exps"), ("w3", "ffn_up_exps")):
        w.add_tensor(f"blk.{l}.{name}.weight", np.stack([f32(st[f"{p}ffn.experts.{e}.{proj}.weight"]) for e in range(ne)]))
w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
print("wrote", out)
