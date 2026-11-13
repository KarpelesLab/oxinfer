#!/usr/bin/env python3
# Writes a (tiny) DeepSeek V4.1 checkpoint in DeepSeek's naming (tools/deepseek41_tiny.py) as a
# `deepseek41` GGUF, with the tensor names and metadata keys of the released conversions (read
# from their headers), to test oxinfer's reader without their hundreds of GB:
#   --style llama (default): llama.cpp's layout (unsloth, ggml-org, ISTA-DASLab): the tables of
#     all engram layers in one `per_layer_token_embd.weight`, the product of the engram gate's
#     two weights as `blk.N.engram_gate`, the hash inputs in the header (`ple.*`,
#     `engram.token_map`, offsets counted over the table of all layers), `wo_a` as a matrix.
#   --style buffers: the layout of AtomicChat's files: the reference's hash buffers as tensors
#     (`engram_token_map`, `engram_primes`, `engram_offsets`, `engram_multipliers`), the gate's
#     weights apart (`engram_q`, `engram_k`), `wo_a` per group, the config's own key names, and
#     the engram tables in a file of their own named by `deepseek41.engram.file` (OUT's folder).
#   --split-engram (llama style): the engram tables as an extra last part that is not named as
#     one (ISTA-DASLab's `DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf`); OUT must then be named
#     `...-00001-of-00002.gguf`.
# Tensors keep their precision (float32 / bf16); no tokenizer is written (the result is for
# examples/deepseek4_check, not for chatting).
# usage: deepseek41_gguf.py MODEL_DIR INFERENCE_DIR OUT.gguf [--style llama|buffers] [--split-engram]
import json, os, re, struct, sys

import numpy as np
import torch
from safetensors.torch import load_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ref_deepseek41 import model_args, reference

d, inference, out = sys.argv[1], sys.argv[2], sys.argv[3]
style = sys.argv[sys.argv.index("--style") + 1] if "--style" in sys.argv else "llama"
split_engram = "--split-engram" in sys.argv
assert style in ("llama", "buffers")
cfg = json.load(open(f"{d}/config.json"))
t = cfg["text_config"]
sd = load_file(f"{d}/model.safetensors")
A = "deepseek41"

U32, I32, F32, BOOL, STR, ARR, U64, I64 = 4, 5, 6, 7, 8, 9, 10, 11
GGML = {torch.float32: 0, torch.bfloat16: 30, torch.int32: 26, torch.int64: 27}


def s(x):
    b = x.encode()
    return struct.pack("<Q", len(b)) + b


def value(ty, v):
    if ty == STR:
        return s(v)
    return struct.pack({U32: "<I", I32: "<i", F32: "<f", BOOL: "<?", U64: "<Q", I64: "<q"}[ty], v)


def kv(key, ty, v, elem=None):
    if ty == ARR:
        body = struct.pack("<IQ", elem, len(v)) + b"".join(value(elem, x) for x in v)
    else:
        body = value(ty, v)
    return s(key) + struct.pack("<I", ty) + body


def write(path, kvs, tensors):
    """`tensors`: name -> torch tensor (written in its dtype, ggml dims = reversed shape)."""
    infos, data = [], b""
    for name, x in tensors.items():
        x = x.contiguous()
        raw = x.view(torch.int16).numpy().tobytes() if x.dtype == torch.bfloat16 else x.numpy().tobytes()
        data += b"\0" * (-len(data) % 32)
        dims = list(reversed(x.shape))
        infos.append(s(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", n) for n in dims)
                     + struct.pack("<IQ", GGML[x.dtype], len(data)))
        data += raw
    head = b"GGUF" + struct.pack("<IQQ", 3, len(infos), len(kvs)) + b"".join(kvs) + b"".join(infos)
    with open(path, "wb") as f:
        f.write(head + b"\0" * (-len(head) % 32) + data)


LAYER = {
    "attn_norm.weight": "attn_norm.weight", "ffn_norm.weight": "ffn_norm.weight",
    "hc_attn_fn": "hc_attn_fn.weight", "hc_attn_base": "hc_attn_base.weight", "hc_attn_scale": "hc_attn_scale.weight",
    "hc_ffn_fn": "hc_ffn_fn.weight", "hc_ffn_base": "hc_ffn_base.weight", "hc_ffn_scale": "hc_ffn_scale.weight",
    "attn.attn_sink": "attn_sinks.weight", "attn.wq_a.weight": "attn_q_a.weight",
    "attn.q_norm.weight": "attn_q_a_norm.weight", "attn.wq_b.weight": "attn_q_b.weight",
    "attn.wkv.weight": "attn_kv.weight", "attn.kv_norm.weight": "attn_kv_a_norm.weight",
    "attn.wo_a.weight": "attn_output_a.weight", "attn.wo_b.weight": "attn_output_b.weight",
    "attn.compressor.wkv.weight": "attn_compressor_kv.weight",
    "attn.compressor.wgate.weight": "attn_compressor_gate.weight",
    "attn.compressor.norm.weight": "attn_compressor_norm.weight",
    "attn.indexer.wq_b.weight": "indexer.attn_q_b.weight", "attn.indexer.weights_proj.weight": "indexer.proj.weight",
    "attn.indexer.wk.weight": "indexer.attn_k.weight", "attn.indexer.k_norm.weight": "indexer.k_norm.weight",
    "ffn.gate.weight": "ffn_gate_inp.weight", "ffn.gate.bias": "exp_probs_b.bias",
    "ffn.gate.bias_vl": "exp_probs_b_vl.bias",
    "ffn.shared_experts.w1.weight": "ffn_gate_shexp.weight", "ffn.shared_experts.w3.weight": "ffn_up_shexp.weight",
    "ffn.shared_experts.w2.weight": "ffn_down_shexp.weight",
    "engram.wkv.weight": "engram_kv.weight" if style == "llama" else "engram_wkv.weight",
    "engram.q_weight": "engram_q.weight", "engram.k_weight": "engram_k.weight",
}
EXPERT = {"w1": "ffn_gate_exps.weight", "w3": "ffn_up_exps.weight", "w2": "ffn_down_exps.weight"}
tensors, tables, experts = {}, {}, {}
for k, v in sd.items():
    if k in ("embed.weight", "head.weight", "norm.weight"):
        tensors[{"embed.weight": "token_embd.weight", "head.weight": "output.weight",
                 "norm.weight": "output_norm.weight"}[k]] = v
        continue
    m = re.fullmatch(r"layers\.(\d+)\.(.*)", k)
    if m is None:
        continue  # engram.* (below), the vision tower (not part of this file)
    l, rest = int(m.group(1)), m.group(2)
    e = re.fullmatch(r"ffn\.experts\.(\d+)\.(w[123])\.weight", rest)
    if e:
        experts.setdefault((l, EXPERT[e.group(2)]), {})[int(e.group(1))] = v
    elif rest == "engram.embed.weight":
        tables[l] = v
    elif rest == "attn.wo_a.weight" and style == "buffers":
        g = t["o_groups"]
        tensors[f"blk.{l}.attn_output_a.weight"] = v.reshape(g, v.shape[0] // g, v.shape[1])
    else:
        tensors[f"blk.{l}.{LAYER[rest]}"] = v
for (l, name), per in experts.items():
    tensors[f"blk.{l}.{name}"] = torch.stack([per[i] for i in range(len(per))])
elayers = t.get("engram_layer_ids", [])
if style == "llama":
    for l in elayers:
        q, k = tensors.pop(f"blk.{l}.engram_q.weight"), tensors.pop(f"blk.{l}.engram_k.weight")
        tensors[f"blk.{l}.engram_gate.weight"] = q.float() * k.float()
table = {"per_layer_token_embd.weight": torch.cat([tables[l] for l in elayers])} if elayers else {}

rs = t.get("rope_scaling") or {}
n = t["num_hidden_layers"]
kvs = [
    kv("general.architecture", STR, A), kv("general.name", STR, "tiny DeepSeek V4.1"),
    kv(f"{A}.block_count", U32, n), kv(f"{A}.context_length", U32, t["max_position_embeddings"]),
    kv(f"{A}.embedding_length", U32, t["hidden_size"]), kv(f"{A}.attention.head_count", U32, t["num_attention_heads"]),
    kv(f"{A}.attention.head_count_kv", U32, 1), kv(f"{A}.rope.scaling.type", STR, "yarn"),
    kv(f"{A}.rope.scaling.factor", F32, rs["factor"]),
    kv(f"{A}.rope.scaling.original_context_length", U32, rs["original_max_position_embeddings"]),
    kv(f"{A}.rope.scaling.yarn_beta_fast", F32, rs["beta_fast"]),
    kv(f"{A}.rope.scaling.yarn_beta_slow", F32, rs["beta_slow"]), kv(f"{A}.rope.freq_base", F32, t["rope_theta"]),
    kv(f"{A}.attention.layer_norm_rms_epsilon", F32, t["rms_norm_eps"]),
    kv(f"{A}.expert_count", U32, t["n_routed_experts"]), kv(f"{A}.expert_used_count", U32, t["num_experts_per_tok"]),
    kv(f"{A}.expert_gating_func", U32, 4), kv(f"{A}.attention.key_length", U32, t["head_dim"]),
    kv(f"{A}.attention.value_length", U32, t["head_dim"]), kv(f"{A}.rope.dimension_count", U32, t["qk_rope_head_dim"]),
    kv(f"{A}.attention.q_lora_rank", U32, t["q_lora_rank"]),
    kv(f"{A}.attention.sliding_window", U32, t["sliding_window"]),
    kv(f"{A}.expert_feed_forward_length", U32, t["moe_intermediate_size"]),
    kv(f"{A}.expert_shared_count", U32, t["n_shared_experts"]),
    kv(f"{A}.expert_weights_scale", F32, t["routed_scaling_factor"]),
    kv(f"{A}.expert_weights_norm", BOOL, t["norm_topk_prob"]),
    kv(f"{A}.swiglu_clamp_exp", ARR, [t["swiglu_limit"]] * n, F32),
    kv(f"{A}.swiglu_clamp_shexp", ARR, [t["swiglu_limit"]] * n, F32),
    kv(f"{A}.attention.indexer.head_count", U32, t["index_n_heads"]),
    kv(f"{A}.attention.indexer.key_length", U32, t["index_head_dim"]),
    kv(f"{A}.attention.indexer.top_k", U32, t["index_topk"]),
    kv(f"{A}.attention.output_group_count", U32, t["o_groups"]),
    kv(f"{A}.attention.output_lora_rank", U32, t["o_lora_rank"]),
    kv(f"{A}.attention.compress_ratios", ARR, t["compress_ratios"], U32),
    kv(f"{A}.attention.compress_rope_freq_base", F32, t["compress_rope_theta"]),
    kv(f"{A}.hyper_connection.count", U32, t["hc_mult"]),
    kv(f"{A}.hyper_connection.sinkhorn_iterations", U32, t["hc_sinkhorn_iters"]),
    kv(f"{A}.hyper_connection.epsilon", F32, t["hc_eps"]), kv(f"{A}.hash_layer_count", U32, 0),
]
ngram, heads = t.get("engram_max_ngram_size", 1), t.get("engram_n_heads", 0)
if style == "llama":
    kvs += [
        kv(f"{A}.attention.kv_source_layers", ARR, t["kv_source_layer_ids"], U32),
        kv(f"{A}.attention.indexer.source_layers", ARR, t["index_source_layer_ids"], U32),
        kv(f"{A}.attention.indexer.candidate_source_layer", I32, t["candidate_source_layer_id"]),
        kv(f"{A}.attention.indexer.candidate_top_k_blocks", U32, t["candidate_topk_blocks"]),
        kv(f"{A}.attention.indexer.candidate_block_size", U32, t["candidate_block_size"]),
    ]
    if elayers:
        M = reference(inference)
        layout = M.EngramLayout.from_args(model_args(M, cfg, 1024))
        primes = [p for layer in layout.primes for per in layer for p in per]
        offsets = np.cumsum([0, *primes[:-1]]).tolist()  # over the table of all layers
        kvs += [
            kv(f"{A}.ple.layers", ARR, elayers, U32), kv(f"{A}.ple.ngram_size", U32, ngram),
            kv(f"{A}.ple.heads_per_ngram", U32, heads),
            kv(f"{A}.embedding_length_per_layer_input", U32, t["engram_head_dim"]),
            kv(f"{A}.ple.layer_multipliers", ARR, sd["engram.multipliers"].flatten().tolist(), U64),
            kv(f"{A}.ple.head_offsets", ARR, offsets, U64), kv(f"{A}.ple.head_vocab_sizes", ARR, primes, U64),
            kv(f"{A}.engram.vocab_size", U32, t["engram_vocab_size"]),
            kv(f"{A}.engram.pad_token_id", U32, t["engram_pad_token_id"]),
            kv(f"{A}.engram.compressed_vocab_size", U32, t["engram_compressed_vocab_size"]),
            kv(f"{A}.engram.token_map", ARR, sd["engram.token_map"].tolist(), I32),
        ]
else:
    kvs += [
        kv(f"{A}.attention.candidate_block_size", U32, t["candidate_block_size"]),
        kv(f"{A}.attention.candidate_topk_blocks", U32, t["candidate_topk_blocks"]),
        kv(f"{A}.attention.candidate_source_layer_id", I32, t["candidate_source_layer_id"]),
        kv(f"{A}.kv_source_layer_ids", ARR, t["kv_source_layer_ids"], U32),
        kv(f"{A}.index_source_layer_ids", ARR, t["index_source_layer_ids"], U32),
        kv(f"{A}.expert_scoring_func", STR, t["scoring_func"]),
    ]
    if elayers:
        M = reference(inference)
        layout = M.EngramLayout.from_args(model_args(M, cfg, 1024))
        primes = torch.tensor([[p for per in layer for p in per] for layer in layout.primes])
        offsets = torch.tensor(np.array([np.cumsum([0, *row[:-1]]) for row in primes.tolist()]))
        tensors = {
            "engram_token_map.weight": sd["engram.token_map"].to(torch.int32), "engram_primes.weight": primes,
            "engram_offsets.weight": offsets, "engram_multipliers.weight": sd["engram.multipliers"], **tensors,
        }
        engram_file = os.path.basename(out).replace(".gguf", "") + "-engram.gguf"
        kvs += [
            kv(f"{A}.engram.present", BOOL, True), kv(f"{A}.engram.layer_ids", ARR, elayers, U32),
            kv(f"{A}.engram.num_embeddings", ARR, t["engram_num_embeddings"], U64),
            kv(f"{A}.engram_head_dim", U32, t["engram_head_dim"]), kv(f"{A}.engram_n_heads", U32, heads),
            kv(f"{A}.engram_max_ngram_size", U32, ngram), kv(f"{A}.engram_vocab_size", U32, t["engram_vocab_size"]),
            kv(f"{A}.engram_compressed_vocab_size", U32, t["engram_compressed_vocab_size"]),
            kv(f"{A}.engram_pad_token_id", U32, t["engram_pad_token_id"]), kv(f"{A}.engram.file", STR, engram_file),
        ]
        write(os.path.join(os.path.dirname(out), engram_file),
              [kv("general.name", STR, "tiny DeepSeek V4.1 engram tables")], table)
        table = {}

if split_engram:
    assert style == "llama" and out.endswith("-00001-of-00002.gguf"), "name OUT ...-00001-of-00002.gguf"
    parts = [kv("split.no", U32, 0), kv("split.count", U32, 2), kv("split.tensors.count", U32, len(tensors) + 1)]
    write(out, kvs + parts, tensors)
    write(os.path.join(os.path.dirname(out), "tiny-Engram.gguf"),
          [kv("general.name", STR, "tiny DeepSeek V4.1 engram tables"), kv("split.no", U32, 1),
           kv("split.count", U32, 2), kv("split.tensors.count", U32, len(tensors) + 1)], table)
else:
    write(out, kvs, {**tensors, **table})
print("saved", out, len(tensors) + len(table), "tensors,", style, "layout")
