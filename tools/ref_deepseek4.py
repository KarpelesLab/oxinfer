#!/usr/bin/env python3
# Reference outputs of a (tiny) DeepSeek V4 checkpoint from DeepSeek's own inference code
# (`inference/model.py` of deepseek-ai/DeepSeek-V4-Flash), run in float32 on the CPU: its TileLang
# kernels are replaced by plain PyTorch equivalents (sparse attention with sinks, the Sinkhorn
# split of the hyper-connections), and the quantization-aware-training simulations (FP8 / FP4
# rounding of the cached keys and the indexer, Hadamard rotation before them) are left out, as
# transformers and llama.cpp do (the rotation alone changes no dot product).
#   prefill: logits at every position, the lightning indexer's top-k of every CSA layer;
#   decode: greedy steps (model.py's own decode path, one token at a time), logits at each step.
# usage: ref_deepseek4.py MODEL_DIR MODEL_PY OUT.safetensors N_PROMPT N_DECODE [SEED] [--chunk N]
#   MODEL_PY: the checkpoint's inference/model.py. --chunk: the prompt as one prefill of N tokens
#   followed by one-token steps (model.py only prefills from position 0).
import importlib.util, json, math, sys, types

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

d, model_py, out, n, nd = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
seed = int(sys.argv[6]) if len(sys.argv) > 6 and not sys.argv[6].startswith("--") else 0
chunk = int(sys.argv[sys.argv.index("--chunk") + 1]) if "--chunk" in sys.argv else n
torch.set_num_threads(8)
torch.set_default_dtype(torch.float32)

kernel = types.ModuleType("kernel")


def act_quant(x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False):
    assert inplace, "only the in-place (QAT simulation) form is reached with float weights"
    return x


def fp4_act_quant(x, block_size=32, inplace=False):
    assert inplace
    return x


def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
    # q [b, m, h, d], kv [b, n, d], topk_idxs [b, m, k] (-1 = none) -> [b, m, h, d]
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
    comb = mixes[..., 2 * hc:] * hc_scale[2] + hc_base[2 * hc:]
    comb = comb.unflatten(-1, (hc, hc))
    comb = torch.softmax(comb, dim=-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb


def unused(*a, **k):
    raise RuntimeError("quantized GEMM reached with float weights")


kernel.act_quant, kernel.fp4_act_quant, kernel.sparse_attn = act_quant, fp4_act_quant, sparse_attn
kernel.hc_split_sinkhorn, kernel.fp8_gemm, kernel.fp4_gemm = hc_split_sinkhorn, unused, unused
sys.modules["kernel"] = kernel
spec = importlib.util.spec_from_file_location("dsv4_model", model_py)
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)
M.rotate_activation = lambda x: x  # orthogonal: no score changes

cfg = json.load(open(f"{d}/config.json"))
rs = cfg.get("rope_scaling") or {}
n_layers = cfg["num_hidden_layers"]
args = M.ModelArgs(
    max_batch_size=1, max_seq_len=max(1024, n + nd + 8), dtype="bf16", scale_fmt=None, expert_dtype=None,
    scale_dtype="fp32", vocab_size=cfg["vocab_size"], dim=cfg["hidden_size"],
    moe_inter_dim=cfg["moe_intermediate_size"], n_layers=n_layers, n_hash_layers=cfg.get("num_hash_layers", 0),
    n_mtp_layers=0, n_heads=cfg["num_attention_heads"], n_routed_experts=cfg["n_routed_experts"],
    n_shared_experts=cfg["n_shared_experts"], n_activated_experts=cfg["num_experts_per_tok"],
    score_func=cfg["scoring_func"], route_scale=cfg["routed_scaling_factor"], swiglu_limit=cfg["swiglu_limit"],
    q_lora_rank=cfg["q_lora_rank"], head_dim=cfg["head_dim"], rope_head_dim=cfg["qk_rope_head_dim"],
    norm_eps=cfg["rms_norm_eps"], o_groups=cfg["o_groups"], o_lora_rank=cfg["o_lora_rank"],
    window_size=cfg["sliding_window"], compress_ratios=tuple(cfg["compress_ratios"]),
    compress_rope_theta=cfg["compress_rope_theta"], original_seq_len=rs.get("original_max_position_embeddings", 0),
    rope_theta=cfg["rope_theta"], rope_factor=rs.get("factor", 1), beta_fast=rs.get("beta_fast", 32),
    beta_slow=rs.get("beta_slow", 1), index_n_heads=cfg["index_n_heads"], index_head_dim=cfg["index_head_dim"],
    index_topk=cfg["index_topk"], hc_mult=cfg["hc_mult"], hc_sinkhorn_iters=cfg["hc_sinkhorn_iters"],
    hc_eps=cfg["hc_eps"],
)
model = M.Transformer(args)
model.float()
torch.set_default_dtype(torch.float32)
sd = load_file(f"{d}/model.safetensors")
sd = {k: (v.float() if v.is_floating_point() else v) for k, v in sd.items()}
missing, unexpected = model.load_state_dict(sd, strict=False)
missing = [k for k in missing if not k.endswith(("kv_cache", "kv_state", "score_state", "freqs_cis"))]
assert not missing and not unexpected, (missing, unexpected)

g = torch.Generator().manual_seed(seed)
ids = torch.cat([torch.tensor([0]), torch.randint(1000, 128000, (n - 1,), generator=g)]).unsqueeze(0)

topk = {}
for i, layer in enumerate(model.layers):
    ix = getattr(layer.attn, "indexer", None)
    if ix is not None:
        def hook(mod, inp, outp, i=i):
            topk.setdefault(i, []).append(outp.detach().clone())
        ix.register_forward_hook(hook)

# model.py returns the last position's logits: run the head on every position of the prefill.
head_all = {}


def head_hook(mod, inp, outp):
    x, fn, scale, base, norm = inp
    head_all["x"] = F.linear(norm(mod.hc_head(x, fn, scale, base)).float(), mod.weight)


model.head.register_forward_hook(head_hook)
hidden = {}
if "--hidden" in sys.argv:
    # The residual (mean of the hc copies) after each layer, for the first (prefill) pass.
    for i, layer in enumerate(model.layers):
        def lhook(mod, inp, outp, i=i):
            hidden.setdefault(i, outp.detach().mean(dim=2)[0].clone())
        layer.register_forward_hook(lhook)
with torch.no_grad():
    logits = []
    model(ids[:, :chunk], 0)
    logits.append(head_all["x"][0])
    for p in range(chunk, n):
        model(ids[:, p:p + 1], p)
        logits.append(head_all["x"][0])
res = {"ids": ids[0].to(torch.int64), "logits": torch.cat(logits).float()}
for i, v in topk.items():
    t = v[0][0].to(torch.int64)
    res[f"topk.{i}"] = torch.sort(t, dim=-1).values
topk.clear()
for i, h in hidden.items():
    res[f"hidden.{i}"] = h.float()

with torch.no_grad():
    nxt = res["logits"][-1].argmax()
    gen, steps = [], []
    for s in range(nd):
        gen.append(int(nxt))
        lg = model(nxt.view(1, 1), n + s)[0]
        steps.append(lg.float())
        nxt = lg.argmax()
res["gen"] = torch.tensor(gen, dtype=torch.int64)
if steps:
    res["decode_logits"] = torch.stack(steps)
save_file({k: v.contiguous() for k, v in res.items()}, out)
print("prompt", ids.shape[1], "gen", gen)
print("top5 last", torch.topk(res["logits"][-1], 5))
