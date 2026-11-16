#!/usr/bin/env python3
"""Writes the language model of a Qwen3-VL checkpoint folder (tools/minimax_tiny.py --arch
qwen3vl) as a llama.cpp GGUF file with the metadata keys and tensor names of the published
`qwen3vl` files (unsloth/Qwen3-VL-32B-Instruct-GGUF: `blk.N.attn_q`, `attn_q_norm`, `ffn_gate`,
..., stored as in the Hugging Face checkpoint; the vision tower is not part of such a file).
float32 tensors: examples/qwen4exp_check.rs must print the same numbers for the GGUF as for the
folder.

usage: qwen3vl_gguf.py MODEL_DIR OUT.gguf        (needs the `gguf` package)
then:  qwen4exp_check OUT.gguf DIR/prompt.txt DIR/ref.bin DIR/gen.txt --cpu
"""
import json, os, sys

import gguf
import numpy as np
from safetensors.numpy import load_file

src, out = sys.argv[1], sys.argv[2]
cfg = json.load(open(os.path.join(src, "config.json")))
t = cfg["text_config"]
sd = load_file(os.path.join(src, "model.safetensors"))
P = next(k for k in sd if k.endswith("embed_tokens.weight"))[: -len("embed_tokens.weight")]
rope = t.get("rope_parameters") or t.get("rope_scaling") or {}
arch = "qwen3vl"
w = gguf.GGUFWriter(out, arch)
k = lambda s: f"{arch}.{s}"
w.add_uint32(k("block_count"), t["num_hidden_layers"])
w.add_uint32(k("context_length"), t.get("max_position_embeddings", 262144))
w.add_uint32(k("embedding_length"), t["hidden_size"])
w.add_uint32(k("feed_forward_length"), t["intermediate_size"])
w.add_uint32(k("attention.head_count"), t["num_attention_heads"])
w.add_uint32(k("attention.head_count_kv"), t["num_key_value_heads"])
w.add_array(k("rope.dimension_sections"), list(rope.get("mrope_section", [24, 20, 20])) + [0])
w.add_float32(k("rope.freq_base"), float(t.get("rope_theta") or rope["rope_theta"]))
w.add_float32(k("attention.layer_norm_rms_epsilon"), t["rms_norm_eps"])
w.add_uint32(k("attention.key_length"), t["head_dim"])
w.add_uint32(k("attention.value_length"), t["head_dim"])
w.add_uint32(k("n_deepstack_layers"), len(cfg.get("vision_config", {}).get("deepstack_visual_indexes", [])))


def put(name, a):
    w.add_tensor(name, np.ascontiguousarray(a, dtype=np.float32))


put("output.weight", sd["lm_head.weight"])
put("output_norm.weight", sd[P + "norm.weight"])
put("token_embd.weight", sd[P + "embed_tokens.weight"])
A = {
    "self_attn.k_proj.weight": "attn_k.weight", "self_attn.k_norm.weight": "attn_k_norm.weight",
    "input_layernorm.weight": "attn_norm.weight", "self_attn.o_proj.weight": "attn_output.weight",
    "self_attn.q_proj.weight": "attn_q.weight", "self_attn.q_norm.weight": "attn_q_norm.weight",
    "self_attn.v_proj.weight": "attn_v.weight", "mlp.down_proj.weight": "ffn_down.weight",
    "mlp.gate_proj.weight": "ffn_gate.weight", "post_attention_layernorm.weight": "ffn_norm.weight",
    "mlp.up_proj.weight": "ffn_up.weight",
}
for l in range(t["num_hidden_layers"]):
    for h, g in A.items():
        put(f"blk.{l}.{g}", sd[f"{P}layers.{l}.{h}"])
w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
print("wrote", out)
