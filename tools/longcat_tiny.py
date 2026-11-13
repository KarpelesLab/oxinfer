#!/usr/bin/env python3
# Builds a tiny random LongCat checkpoint in the layout of meituan-longcat/LongCat-2.5 (config.json
# with its keys, `model.safetensors` with its tensor names and dtypes: one module per expert,
# `model.oe_embed_tokens{i}` / `model.oe_embed_proj{i}`, the router and `weights_proj` in
# float32, the rest in bf16), with every feature of the text model switched on: two sublayers
# and a shortcut MoE per layer, MLA with both LoRA scales and DeepSeek's YaRN, identity ("zero")
# experts, the n-gram embedding with tokens left out of it (`oe_ignore_tokens`), and an indexer
# on every second attention whose top-k is small enough that most queries are sparse, with keys
# always attended at both ends.
#
# usage: longcat_tiny.py OUT_DIR [--flash | --ngram] [--tokenizer DIR] [--vocab N]
#   --flash: LongCat-Flash-Chat's architecture instead (`LongcatFlashForCausalLM`: no indexer,
#   no n-gram embedding, plain RoPE). --ngram: LongCat-Flash-Lite's (`LongcatFlashNgram
#   ForCausalLM`: the n-gram embedding under its names, YaRN). Both run on transformers and
#   Meituan's code as published.
#   --tokenizer: copy tokenizer.json, tokenizer_config.json and chat_template.jinja from DIR.
#   --vocab: the vocabulary size (default 1024; 163840 with LongCat-2.5's tokenizer, to run the
#   checkpoint from the command line).
# Then: ref_longcat.py OUT_DIR CODE_DIR ref.safetensors 96 16, and
#   cargo run --release --example longcat_check -- OUT_DIR ref.safetensors [--chunk N]
import json, math, os, shutil, sys

import torch
from safetensors.torch import save_file

out = sys.argv[1]
flash, ngram = "--flash" in sys.argv, "--ngram" in sys.argv
os.makedirs(out, exist_ok=True)
if "--tokenizer" in sys.argv:
    src = sys.argv[sys.argv.index("--tokenizer") + 1]
    for fn in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "special_tokens_map.json"):
        if os.path.exists(os.path.join(src, fn)):
            shutil.copy(os.path.join(src, fn), os.path.join(out, fn))

H, L, NH = 64, 3, 4
V = int(sys.argv[sys.argv.index("--vocab") + 1]) if "--vocab" in sys.argv else 1024
QL, LAT, DN, DR, VD = 24, 16, 16, 8, 16
FFN, EFF, NE, NZ, TOPK = 96, 48, 6, 3, 3
cfg = {
    "architectures": ["LongcatCausalLM"], "model_type": "pro_v", "attention_bias": False, "attention_dropout": 0.0,
    "vocab_size": V, "hidden_size": H, "ffn_hidden_size": FFN, "expert_ffn_hidden_size": EFF, "num_layers": L,
    "num_attention_heads": NH, "kv_lora_rank": LAT, "q_lora_rank": QL, "qk_rope_head_dim": DR, "v_head_dim": VD,
    "qk_nope_head_dim": DN, "mla_scale_q_lora": True, "mla_scale_kv_lora": True, "routed_scaling_factor": 3,
    "n_routed_experts": NE, "max_position_embeddings": 4096, "rms_norm_eps": 1e-05, "use_cache": True,
    "bos_token_id": 1, "eos_token_id": 2, "rope_theta": 10000.0, "attention_method": "MLA",
    "zero_expert_num": NZ, "zero_expert_type": "identity", "moe_topk": TOPK, "use_mla": 1,
    "moe_switch_token_num": 1024, "moe_impl": "mix",
    "oe_vocab_size_ratio": 3.567, "oe_neighbor_num": 5, "oe_split_num": 4, "oe_ignore_tokens": "199:224",
    "index_n_heads": 4, "index_head_dim": 16, "index_topk": 12, "index_dtype": "bf16", "index_k_norm_type": "rms",
    "cli_factor": 2, "index_local_tokens": 3, "index_init_tokens": 2,
    "rope_scaling": {"original_max_position_embeddings": 32, "rope_type": "deepseek_yarn", "factor": 12,
                     "beta_fast": 32, "beta_slow": 1, "mscale": 1, "mscale_all_dim": 1},
}
if flash or ngram:
    for k in [k for k in cfg if k.startswith(("index_", "oe_", "cli_", "moe_switch", "moe_impl", "use_mla"))]:
        del cfg[k]
    del cfg["model_type"]
    cfg["architectures"] = ["LongcatFlashNgramForCausalLM" if ngram else "LongcatFlashForCausalLM"]
if flash:
    del cfg["rope_scaling"]
if ngram:
    cfg.update({"ngram_vocab_size_ratio": 3.567, "emb_neighbor_num": 4, "emb_split_num": 4})
    cfg["rope_scaling"]["rope_type"] = "yarn"

g = torch.Generator().manual_seed(25)


def rn(*shape):
    return torch.randn(*shape, generator=g)


tensors = {}


def mat(name, n_out, n_in, gain=1.0, f32=False):
    v = gain * rn(n_out, n_in) / math.sqrt(n_in)
    tensors[name] = v if f32 else v.to(torch.bfloat16)


def norm(name, n):
    tensors[name] = (1.0 + 0.2 * rn(n)).to(torch.bfloat16)


# Gains near one keep the model well conditioned (see tools/kimi_k31_tiny.py).
mat("model.embed_tokens.weight", V, H, gain=math.sqrt(H))
mat("lm_head.weight", V, H, gain=2.0)
norm("model.norm.weight", H)
if "oe_vocab_size_ratio" in cfg or ngram:
    ratio = cfg.get("oe_vocab_size_ratio", cfg.get("ngram_vocab_size_ratio"))
    n = cfg.get("oe_neighbor_num", cfg.get("emb_neighbor_num"))
    k = cfg.get("oe_split_num", cfg.get("emb_split_num"))
    m, nt = int(V * ratio), (n - 1) * k
    for i in range(nt):
        tn, pn = (f"model.ngram_embeddings.embedders.{i}.weight", f"model.ngram_embeddings.post_projs.{i}.weight") if ngram else (
            f"model.oe_embed_tokens{i}.weight", f"model.oe_embed_proj{i}.weight")
        tensors[tn] = rn(m + 2 * i + 1, H // nt).to(torch.bfloat16)
        mat(pn, H, H // nt)
for l in range(L):
    lp = f"model.layers.{l}."
    for i in range(2):
        norm(f"{lp}input_layernorm.{i}.weight", H)
        norm(f"{lp}post_attention_layernorm.{i}.weight", H)
        ap = f"{lp}self_attn.{i}."
        mat(ap + "q_a_proj.weight", QL, H)
        norm(ap + "q_a_layernorm.weight", QL)
        # (The LoRA scales multiply queries and keys by sqrt(H / rank): smaller gains keep the
        # attention logits in a usual range.)
        mat(ap + "q_b_proj.weight", NH * (DN + DR), QL, gain=0.7)
        mat(ap + "kv_a_proj_with_mqa.weight", LAT + DR, H)
        norm(ap + "kv_a_layernorm.weight", LAT)
        mat(ap + "kv_b_proj.weight", NH * (DN + VD), LAT, gain=0.5)
        mat(ap + "o_proj.weight", H, NH * VD, gain=0.5)
        if "index_topk" in cfg and (2 * l + i) % cfg["cli_factor"] == 0:
            ni, d = cfg["index_n_heads"], cfg["index_head_dim"]
            mat(ap + "indexer.wq_b.weight", ni * d, QL, gain=2.0)
            mat(ap + "indexer.wk.weight", d, H, gain=2.0)
            norm(ap + "indexer.k_norm.weight", d)
            mat(ap + "indexer.weights_proj.weight", ni, H, f32=True)
        mp = f"{lp}mlps.{i}."
        mat(mp + "gate_proj.weight", FFN, H)
        mat(mp + "up_proj.weight", FFN, H)
        mat(mp + "down_proj.weight", H, FFN, gain=0.5)
    mat(f"{lp}mlp.router.classifier.weight", NE + NZ, H, f32=True)
    tensors[f"{lp}mlp.router.e_score_correction_bias"] = 0.05 * rn(NE + NZ)
    for e in range(NE):
        ep = f"{lp}mlp.experts.{e}."
        mat(ep + "gate_proj.weight", EFF, H)
        mat(ep + "up_proj.weight", EFF, H)
        mat(ep + "down_proj.weight", H, EFF, gain=0.5)

save_file({k: v.contiguous() for k, v in tensors.items()}, os.path.join(out, "model.safetensors"), metadata={"format": "pt"})
with open(os.path.join(out, "config.json"), "w") as f:
    json.dump(cfg, f, indent=2)
print("saved", out, len(tensors), "tensors")
