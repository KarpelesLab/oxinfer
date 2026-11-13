"""Writes a tiny Nemotron-H checkpoint (tools/nemotron_tiny.py --layout stacked) as a llama.cpp
GGUF file, with the metadata keys and tensor names of the published `nemotron_h_moe` files
(ggml-org/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF): per-block `feed_forward_length` and
`attention.head_count_kv` (0 where a block has no MLP / no attention), `ssm_a` holds
`-exp(A_log)` as `[heads, 1]`, the convolution kernels as `[channels, kernel]`, the gated norm as
`[groups, width / groups]`, the experts stacked. float32 tensors: for
`examples/nemotron_check.rs`, which must print the same numbers for the GGUF as for the folder.

usage: nemotron_gguf.py MODEL_DIR OUT.gguf [PUBLISHED.gguf]      (needs the `gguf` package)
With a published GGUF (its header is enough), its tokenizer and chat template are copied.
"""
import json, os, sys
import numpy as np
import gguf
from safetensors.numpy import load_file

src, out = sys.argv[1], sys.argv[2]
t = json.load(open(os.path.join(src, "config.json")))
sd = load_file(os.path.join(src, "model.safetensors"))
if "model.embeddings.weight" not in sd:
    raise SystemExit("expected the module's own tensor names (nemotron_tiny.py --layout stacked)")
blocks = t["layers_block_type"]
n = len(blocks)
moe = "moe" in blocks
arch = "nemotron_h_moe" if moe else "nemotron_h"
inner = t["mamba_num_heads"] * t["mamba_head_dim"]

w = gguf.GGUFWriter(out, arch)
k = lambda s: f"{arch}.{s}"
w.add_uint32(k("block_count"), n)
w.add_uint32(k("context_length"), t["max_position_embeddings"])
w.add_uint32(k("embedding_length"), t["hidden_size"])
ff = {"moe": t["moe_intermediate_size"], "mlp": t["intermediate_size"]}
w.add_array(k("feed_forward_length"), [ff.get(b, 0) for b in blocks])
w.add_uint32(k("attention.head_count"), t["num_attention_heads"])
w.add_array(k("attention.head_count_kv"), [t["num_key_value_heads"] if b == "full_attention" else 0 for b in blocks])
w.add_float32(k("rope.freq_base"), 10000.0)
w.add_float32(k("attention.layer_norm_rms_epsilon"), t["layer_norm_epsilon"])
w.add_float32(k("attention.layer_norm_epsilon"), t["layer_norm_epsilon"])
if moe:
    w.add_uint32(k("expert_count"), t["n_routed_experts"])
    w.add_uint32(k("expert_used_count"), t["num_experts_per_tok"])
    w.add_uint32(k("expert_group_count"), 1)
    w.add_uint32(k("expert_group_used_count"), 1)
    w.add_uint32(k("expert_feed_forward_length"), t["moe_intermediate_size"])
    w.add_uint32(k("expert_shared_feed_forward_length"), t["moe_shared_expert_intermediate_size"])
    w.add_uint32(k("expert_shared_count"), 1)
    w.add_bool(k("expert_weights_norm"), True)
    w.add_float32(k("expert_weights_scale"), t["routed_scaling_factor"])
w.add_uint32(k("vocab_size"), t["vocab_size"])
w.add_uint32(k("ssm.conv_kernel"), t["conv_kernel"])
w.add_uint32(k("ssm.state_size"), t["ssm_state_size"])
w.add_uint32(k("ssm.group_count"), t["n_groups"])
w.add_uint32(k("ssm.inner_size"), inner)
w.add_uint32(k("ssm.time_step_rank"), t["mamba_num_heads"])
w.add_uint32(k("attention.key_length"), t["head_dim"])
w.add_uint32(k("attention.value_length"), t["head_dim"])


def put(name, a):
    w.add_tensor(name, np.ascontiguousarray(a, dtype=np.float32))


put("token_embd.weight", sd["model.embeddings.weight"])
put("output.weight", sd["lm_head.weight"])
put("output_norm.weight", sd["model.norm_f.weight"])
A = {
    "norm.weight": "attn_norm.weight",
    "mixer.in_proj.weight": "ssm_in.weight", "mixer.conv1d.bias": "ssm_conv1d.bias", "mixer.dt_bias": "ssm_dt.bias",
    "mixer.out_proj.weight": "ssm_out.weight",
    "mixer.q_proj.weight": "attn_q.weight", "mixer.k_proj.weight": "attn_k.weight",
    "mixer.v_proj.weight": "attn_v.weight", "mixer.o_proj.weight": "attn_output.weight",
    "mixer.gate.weight": "ffn_gate_inp.weight", "mixer.gate.e_score_correction_bias": "exp_probs_b.bias",
    "mixer.experts.up_proj": "ffn_up_exps.weight", "mixer.experts.down_proj": "ffn_down_exps.weight",
    "mixer.shared_experts.up_proj.weight": "ffn_up_shexp.weight",
    "mixer.shared_experts.down_proj.weight": "ffn_down_shexp.weight",
    "mixer.up_proj.weight": "ffn_up.weight", "mixer.down_proj.weight": "ffn_down.weight",
}
for l in range(n):
    lp = f"model.layers.{l}."
    for nm in [x[len(lp):] for x in sd if x.startswith(lp)]:
        a = sd[lp + nm]
        if nm in A:
            put(f"blk.{l}.{A[nm]}", a)
        elif nm == "mixer.A_log":
            put(f"blk.{l}.ssm_a", -np.exp(a).reshape(-1, 1))
        elif nm == "mixer.D":
            put(f"blk.{l}.ssm_d", a.reshape(-1, 1))
        elif nm == "mixer.conv1d.weight":
            put(f"blk.{l}.ssm_conv1d.weight", a.reshape(a.shape[0], a.shape[2]))
        elif nm == "mixer.norm.weight":
            put(f"blk.{l}.ssm_norm.weight", a.reshape(t["n_groups"], -1))
        else:
            raise SystemExit(f"unmapped tensor {lp}{nm}")
if len(sys.argv) > 3:
    # The tokenizer and chat template of a published file (its header is enough).
    r = gguf.GGUFReader(sys.argv[3])
    for name, f in r.fields.items():
        if name.startswith("tokenizer."):
            v = f.contents()
            ty = f.types[0]
            if ty == gguf.GGUFValueType.ARRAY:
                w.add_array(name, v)
            else:
                w.add_key_value(name, v, ty)
w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
print(f"{out}: {arch}, {n} blocks")
