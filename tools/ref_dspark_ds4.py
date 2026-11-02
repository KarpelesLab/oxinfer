#!/usr/bin/env python3
# Reference DeepSeek V4 DSpark draft from DeepSeek's own inference code (the `inference/model.py`
# of deepseek-ai/DeepSeek-V4-Flash-DSpark: `DSparkBlock`, `DSparkAttention`, the Markov head), in
# float32 on the CPU (TileLang kernels replaced by PyTorch, FP8 / FP4 weights dequantized as they
# are used, activation quantization off), run on the target features oxinfer dumped, to check
# oxinfer's draft logits.
#
# usage: ref_dspark_ds4.py CKPT_DIR MODEL_PY DUMP_DIR [--target-gguf PART.gguf GGUF_PY_DIR]
#   CKPT_DIR: config.json and the safetensors holding `mtp.*` (the tiny checkpoint of
#     tools/deepseek4_tiny.py --dspark, or the DSpark shards of the real one); the target's
#     `embed.weight` / `head.weight` come from there too, or from the target GGUF part holding
#     token_embd.weight and output.weight (dequantized with gguf-py).
#   DUMP_DIR: written by `cargo run --release --example dspark_check -- MODEL DRAFT PROMPT DUMP_DIR
#     --k 5`: meta.json {"n", "anchor", "k", "drafts"}, feats.f32 [n, n_feat], logits.f32 [k, vocab].
# Prints per slot: max |diff|, relative error, argmax agreement, KL.
import importlib.util, json, math, os, sys, types

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

ckpt, model_py, dump = sys.argv[1], sys.argv[2], sys.argv[3]
torch.set_num_threads(os.cpu_count())
torch.set_default_dtype(torch.float32)

kernel = types.ModuleType("kernel")


def act_quant(x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False):
    return x  # QAT simulation off (the in-place form); the activation quantization of linear() is bypassed below


def fp4_act_quant(x, block_size=32, inplace=False):
    return x


def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
    b, m, h, dd = q.shape
    idx = topk_idxs.long()
    valid = idx >= 0
    g = torch.gather(kv.unsqueeze(1).expand(b, m, -1, dd), 2, idx.clamp(min=0).unsqueeze(-1).expand(-1, -1, -1, dd))
    s = torch.einsum("bmhd,bmkd->bmhk", q.float(), g.float()) * softmax_scale
    s = s.masked_fill(~valid.unsqueeze(2), float("-inf"))
    sink = attn_sink.float().view(1, 1, h, 1).expand(b, m, h, 1)
    p = torch.softmax(torch.cat([s, sink], dim=-1), dim=-1)[..., :-1]
    return torch.einsum("bmhk,bmkd->bmhd", p, g.float()).to(q.dtype)


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    hc = hc_mult
    pre = torch.sigmoid(mixes[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2 * torch.sigmoid(mixes[..., hc:2 * hc] * hc_scale[1] + hc_base[hc:2 * hc])
    comb = (mixes[..., 2 * hc:] * hc_scale[2] + hc_base[2 * hc:]).unflatten(-1, (hc, hc))
    comb = torch.softmax(comb, dim=-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb


kernel.act_quant, kernel.fp4_act_quant, kernel.sparse_attn, kernel.hc_split_sinkhorn = act_quant, fp4_act_quant, sparse_attn, hc_split_sinkhorn
kernel.fp8_gemm = kernel.fp4_gemm = None
sys.modules["kernel"] = kernel
spec = importlib.util.spec_from_file_location("dspark_model", model_py)
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


def dequant(w):
    """A weight as float32: FP8 with 128 x 128 E8M0 block scales, FP4 (packed in int8, low nibble
    first) with E8M0 scales per 32, or float."""
    s = getattr(w, "scale", None)
    if w.dtype == torch.float8_e4m3fn:
        o, i = w.shape
        sc = s.float().repeat_interleave(128, 0)[:o].repeat_interleave(128, 1)[:, :i]
        return w.float() * sc
    if w.dtype == torch.int8:
        b = w.view(torch.uint8)
        v = torch.stack([E2M1[(b & 15).long()], E2M1[(b >> 4).long()]], dim=-1).flatten(-2)
        return v * s.float().repeat_interleave(32, 1)
    return w.float()


def linear(x, weight, bias=None):
    return F.linear(x.float(), dequant(weight))


M.linear = linear

cfg = json.load(open(f"{ckpt}/config.json"))
rs = cfg.get("rope_scaling") or {}
n_layers = cfg["num_hidden_layers"]
ns = 3
args = M.ModelArgs(
    max_batch_size=1, max_seq_len=8192, dtype="bf16", scale_fmt=None, expert_dtype=None, scale_dtype="fp32",
    vocab_size=cfg["vocab_size"], dim=cfg["hidden_size"], moe_inter_dim=cfg["moe_intermediate_size"],
    n_layers=n_layers, n_hash_layers=cfg.get("num_hash_layers", 0), n_mtp_layers=ns, n_heads=cfg["num_attention_heads"],
    n_routed_experts=cfg["n_routed_experts"], n_shared_experts=cfg["n_shared_experts"],
    n_activated_experts=cfg["num_experts_per_tok"], score_func=cfg["scoring_func"],
    route_scale=cfg["routed_scaling_factor"], swiglu_limit=cfg["swiglu_limit"], q_lora_rank=cfg["q_lora_rank"],
    head_dim=cfg["head_dim"], rope_head_dim=cfg["qk_rope_head_dim"], norm_eps=cfg["rms_norm_eps"],
    o_groups=cfg["o_groups"], o_lora_rank=cfg["o_lora_rank"], window_size=cfg["sliding_window"],
    compress_ratios=tuple(cfg["compress_ratios"]), compress_rope_theta=cfg["compress_rope_theta"],
    original_seq_len=rs.get("original_max_position_embeddings", 0), rope_theta=cfg["rope_theta"],
    rope_factor=rs.get("factor", 1), beta_fast=rs.get("beta_fast", 32), beta_slow=rs.get("beta_slow", 1),
    index_n_heads=cfg["index_n_heads"], index_head_dim=cfg["index_head_dim"], index_topk=cfg["index_topk"],
    hc_mult=cfg["hc_mult"], hc_sinkhorn_iters=cfg["hc_sinkhorn_iters"], hc_eps=cfg["hc_eps"],
    dspark_block_size=cfg["dspark_block_size"], dspark_noise_token_id=cfg["dspark_noise_token_id"],
    dspark_target_layer_ids=tuple(cfg["dspark_target_layer_ids"]), dspark_markov_rank=cfg["dspark_markov_rank"],
)
args.temperature = 0
# The stages alone (their expert weights are replaced as they load, so their bf16 placeholders are
# never touched).
stages = torch.nn.ModuleList([M.DSparkBlock(n_layers + i, args) for i in range(ns)])

# Every tensor of the safetensors files under CKPT_DIR.
files = [f for f in sorted(os.listdir(ckpt)) if f.endswith(".safetensors")]
handles = [safe_open(f"{ckpt}/{f}", "pt") for f in files]
where = {k: h for h in handles for k in h.keys()}


def get(k):
    return where[k].get_tensor(k)


mods = dict(stages.named_modules())
for k in where:
    # (Block scales are read with their weights.)
    if not k.startswith("mtp.") or k.endswith(".scale") and k[: -len(".scale")] + ".weight" in where:
        continue
    rest = k[len("mtp."):]
    mod_name, pname = rest.rsplit(".", 1)
    mod = mods.get(mod_name) if pname in ("weight", "bias") else None
    t = get(k)
    if mod is None:
        # Plain parameters of a stage (hc_*, attn_sink as attn.attn_sink).
        obj = stages
        parts = rest.split(".")
        for p_ in parts[:-1]:
            obj = getattr(obj, p_) if not p_.isdigit() else obj[int(p_)]
        p = getattr(obj, parts[-1])
        p.data = t.float() if t.is_floating_point() else t
        continue
    p = getattr(mod, pname)
    sk = k[: -len(pname)] + "scale"
    if sk in where and t.dtype in (torch.float8_e4m3fn, torch.int8):
        t.scale = get(sk)
        if mod_name.endswith("attn.wo_a"):
            t = dequant(t)  # used as a plain tensor by the reference (its einsum)
    elif t.dtype == torch.bfloat16:
        t = t.float()
    new = torch.nn.Parameter(t, requires_grad=False)
    if hasattr(t, "scale"):
        new.scale = t.scale
    setattr(mod, pname, new)
missing = [n for n, p in stages.named_parameters() if p.dtype == torch.bfloat16 and "experts" not in n]
assert not missing, missing

# The target's embeddings and head.
if "--target-gguf" in sys.argv:
    i = sys.argv.index("--target-gguf")
    sys.path.insert(0, sys.argv[i + 2])
    from gguf import GGUFReader
    from gguf.quants import dequantize
    T = {t.name: t for t in GGUFReader(sys.argv[i + 1]).tensors}

    def W(name):
        t = T[name]
        a = dequantize(t.data, t.tensor_type)
        return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).reshape([int(x) for x in reversed(t.shape)])

    emb, head = W("token_embd.weight"), W("output.weight")
else:
    emb, head = get("embed.weight").float(), get("head.weight").float()
embed = M.ParallelEmbedding(args.vocab_size, args.dim)
embed.weight.data = emb
phead = M.ParallelHead(args.vocab_size, args.dim)
phead.weight.data = head
for s in stages:
    s.embed, s.head = embed, phead

meta = json.load(open(f"{dump}/meta.json"))
n, anchor = meta["n"], meta["anchor"]
feats = torch.from_numpy(np.fromfile(f"{dump}/feats.f32", dtype=np.float32)).view(1, n, -1)
ours = torch.from_numpy(np.fromfile(f"{dump}/logits.f32", dtype=np.float32)).view(meta["k"], -1)
with torch.inference_mode():
    a = torch.tensor([anchor])
    h, main_x = stages[0].forward_embed(feats[:, : n - 1], a)
    for s in stages:
        s(h, 0, a, main_x)
    h, main_x = stages[0].forward_embed(feats[:, n - 1:n], a)
    for s in stages:
        h = s(h, n - 1, a, main_x)
    ids, logits, conf = stages[-1].forward_head(h, a)
ref = logits[0].float()
print("reference drafts", ids[0, 1:].tolist(), "ours", meta["drafts"], "confidence", torch.sigmoid(conf[0]).tolist())
for i in range(meta["k"]):
    x, y = ours[i], ref[i]
    d = (x - y).abs().max().item()
    rel = d / y.abs().max().item()
    lx, ly = F.log_softmax(x.double(), -1), F.log_softmax(y.double(), -1)
    kl = (ly.exp() * (ly - lx)).sum().item()
    print(f"slot {i}: max |diff| {d:.3e} (rel {rel:.2e}), argmax ours {x.argmax().item()} ref {y.argmax().item()}, KL {kl:.2e}")
