"""Reference LoRA gradients of an Inkling-2 loop model from PyTorch autograd on transformers'
`Inkling2LoopForConditionalGeneration`, for `examples/inkling2_train.rs check`.

usage: ref_inkling2_lora_grad.py MODEL_DIR OUT.safetensors [--len N] [--targets K] [--rank R]
                                 [--alpha A] [--proj wo_ud,wq_du,...] [--seed S] [--bstd X]
                                 [--real [--text TEXT]]

A low-rank pair (`lora_A` uniform in +-1/sqrt(n_in), `lora_B` normal of std `--bstd`, so that
both factors have a gradient) is added to the chosen attention projections of every layer as a
forward hook on the module's `nn.Linear`: `y + (alpha / r) (x A^T) B^T`. A looped layer has one
module, so its pair runs in every pass. The loss is over the last K tokens: token j, predicted
from position j - 1, contributes `-w log p` for a weight `w >= 0` and `|w| log(1 - p)` negated
(`-|w| ln(max(1 - p, 1e-4))`) for `w < 0`; every fourth target has weight -0.7.

MODEL_DIR is a tiny checkpoint of `tools/inkling2_tiny.py` (with or without `--loop`; run in
float64, random token ids), or with --real the release (`thinkingmachines/Inkling-2-Mini`): float32 on the weights
exactly as stored, as `tools/ref_inkling2_mini.py` loads them, on a chat prompt (about 56 GB).

Writes `ids`, `tgt_idx` (I64), `tgt_w`, `alpha`, `loss`, `logp` (per target), and per site
`A.{layer}.{name}`, `B.{layer}.{name}`, `gA.{layer}.{name}`, `gB.{layer}.{name}` (float32), with
the release's projection names.
"""
import os, sys
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

from transformers.models.inkling2_loop import modeling_inkling2_loop as mod

model_dir, out = sys.argv[1], sys.argv[2]
a = sys.argv[3:]
plen, ntg, rank, alpha, projs, seed, bstd, real, text = 48, 14, 2, 4.0, "wo_ud", 0, 0.05, False, None
for i, x in enumerate(a):
    if x == "--len":
        plen = int(a[i + 1])
    if x == "--targets":
        ntg = int(a[i + 1])
    if x == "--rank":
        rank = int(a[i + 1])
    if x == "--alpha":
        alpha = float(a[i + 1])
    if x == "--proj":
        projs = a[i + 1]
    if x == "--seed":
        seed = int(a[i + 1])
    if x == "--bstd":
        bstd = float(a[i + 1])
    if x == "--real":
        real = True
    if x == "--text":
        text = a[i + 1]

# The release's name of each projection, and its nn.Linear inside transformers' attention.
MODULES = {
    "wq_du": "q_proj.linear",
    "wk_dv": "k_proj.linear",
    "wv_dv": "v_proj.linear",
    "wr_du": "relative_position_bias.linear",
    "gate_attn": "o_proj.gate_proj",
    "attn_sink_proj": "sink_proj",
    "wo_ud": "o_proj.linear",
}

if real:
    import transformers.core_model_loading as cml
    from transformers import AutoTokenizer

    orig = cml.Inkling2LoopDequantizeExperts.convert

    def convert(self, input_dict, source_patterns, target_patterns, model=None, full_layer_name=None, **kw):
        tensors = {k: v[0] if isinstance(v, list) else v for k, v in input_dict.items()}
        if len(tensors) == 1:
            return orig(self, input_dict, source_patterns, target_patterns, model=model, full_layer_name=full_layer_name, **kw)
        weight, scale, amax, den = (tensors.get(p) for p in source_patterns)
        e, rows = weight.shape[:2]
        cols = weight.shape[2] * 2 if amax is not None else weight.shape[2]
        outw = torch.empty(e, rows, cols, dtype=torch.bfloat16)
        if amax is not None:
            e2m1 = torch.tensor(self.E2M1_VALUES)
            byte = torch.arange(256)
            b2v = torch.stack((e2m1[byte & 0xF], e2m1[byte >> 4]), dim=-1)
        for s in range(0, e, 8):
            sl = slice(s, s + 8)
            bs = scale[sl].float()
            if amax is None:
                v = weight[sl].float().view(*bs.shape, 32) * torch.exp2(bs - 127)[..., None]
            else:
                v = b2v[weight[sl].to(torch.uint8).long()].view(*bs.shape, 16) * bs[..., None]
            outw[sl] = v.reshape(-1, rows, cols).to(torch.bfloat16)
        return {target_patterns[0]: outw}

    cml.Inkling2LoopDequantizeExperts.convert = convert

    def experts_forward(self, hidden_states, top_k_index, top_k_weights):
        final = torch.zeros_like(hidden_states, dtype=torch.float32)
        for e in top_k_index.unique().tolist():
            if e == self.num_experts:
                continue
            pos, tok = torch.where(top_k_index.T == e)
            x = hidden_states[tok].float()
            gu = self.gate_up_proj[e].float() * self.gu_factor[e]
            dn = self.down_proj[e].float() * self.dn_factor[e]
            hcur = F.linear(self._apply_gate(F.linear(x, gu)), dn) * top_k_weights[tok, pos, None]
            final = final.index_add(0, tok, hcur)
        return final.to(hidden_states.dtype)

    mod.Inkling2LoopExperts.forward = experts_forward
    m = mod.Inkling2LoopForConditionalGeneration.from_pretrained(
        model_dir, dtype=torch.bfloat16, attn_implementation="eager", experts_implementation="eager"
    ).eval()
    with torch.no_grad():
        for name, p in m.named_parameters():
            if ".mlp.experts." not in name:
                p.data = p.data.float()
        for name, b in m.named_buffers():
            if b.is_floating_point():
                b.data = b.data.float()
        raw = {}
        for f in os.listdir(model_dir):
            if f.startswith("model") and f.endswith(".safetensors"):
                with safe_open(os.path.join(model_dir, f), "pt") as fh:
                    for k in fh.keys():
                        if k.endswith("_amax") or k.endswith("_outer_scale_denominator"):
                            raw[k] = fh.get_tensor(k).float()
        for name, mm in m.named_modules():
            if isinstance(mm, mod.Inkling2LoopExperts):
                layer = int(name.split("layers.")[1].split(".")[0])
                for leaf, short in (("w13_weight", "gu"), ("w2_weight", "dn")):
                    k = f"model.llm.layers.{layer}.mlp.experts.{leaf}"
                    fac = raw[k + "_amax"] / raw[k + "_outer_scale_denominator"] if k + "_amax" in raw else torch.ones(mm.num_experts)
                    setattr(mm, short + "_factor", fac)
    tk = AutoTokenizer.from_pretrained(model_dir)
    msgs = [{"role": "user", "content": text or "Why is the sky blue? Answer in two sentences."}]
    prompt = tk.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, reasoning_effort=0.0)
    reply = "<|message_model|><|content_text|>Sunlight is scattered by the air, and blue light is scattered the most.<|end_message|>"
    ids = torch.tensor([tk(prompt + reply, add_special_tokens=False)["input_ids"]])
    ntg = min(ntg, ids.shape[1] - 1)
    dt = torch.float32
else:
    import json

    # (A tiny checkpoint without the loop, `inkling2`, runs on transformers' class for it.)
    cls = mod.Inkling2LoopForConditionalGeneration
    if json.load(open(os.path.join(model_dir, "config.json")))["model_type"] == "inkling2":
        from transformers.models.inkling2.modeling_inkling2 import Inkling2ForConditionalGeneration as cls
    m = cls.from_pretrained(model_dir, dtype=torch.float64, attn_implementation="eager", experts_implementation="eager").eval()
    g = torch.Generator().manual_seed(seed + 1)
    ids = torch.randint(8, m.config.text_config.unpadded_vocab_size - 8, (1, plen), generator=g)
    dt = torch.float64
for p in m.parameters():
    p.requires_grad_(False)

g = torch.Generator().manual_seed(seed + 2)
scale = alpha / rank
layers = m.model.language_model.layers
pairs, hooks = {}, []
for l, layer in enumerate(layers):
    for name in projs.split(","):
        lin = layer.self_attn.get_submodule(MODULES[name])
        n_out, n_in = lin.weight.shape
        A = ((torch.rand(rank, n_in, generator=g) * 2 - 1) / n_in**0.5).float()
        B = (torch.randn(n_out, rank, generator=g) * bstd).float()
        A, B = A.to(dt).requires_grad_(True), B.to(dt).requires_grad_(True)
        pairs[(l, name)] = (A, B)

        def hook(module, inp, y, A=A, B=B):
            x = inp[0]
            return y + scale * ((x.to(A.dtype) @ A.T) @ B.T).to(y.dtype)

        hooks.append(lin.register_forward_hook(hook))

n = ids.shape[1]
tgt = list(range(n - ntg, n))
w = torch.tensor([-0.7 if j % 4 == 1 else 1.0 + 0.1 * (j % 3) for j in tgt], dtype=torch.float64)
logits = m(ids, use_cache=False).logits[0].double()
logp_all = torch.log_softmax(logits, dim=-1)
logp = torch.stack([logp_all[j - 1, ids[0, j]] for j in tgt])
pos = -w * logp
neg = w * torch.log((1 - logp.exp()).clamp_min(1e-4))
loss = torch.where(w >= 0, pos, neg).sum()
loss.backward()

res = {
    "ids": ids[0].to(torch.int64),
    "tgt_idx": torch.tensor(tgt, dtype=torch.int64),
    "tgt_w": w.float(),
    "alpha": torch.tensor([alpha], dtype=torch.float32),
    "loss": loss.detach().reshape(1).float(),
    "logp": logp.detach().float(),
}
for (l, name), (A, B) in pairs.items():
    res[f"A.{l}.{name}"] = A.detach().float().contiguous()
    res[f"B.{l}.{name}"] = B.detach().float().contiguous()
    res[f"gA.{l}.{name}"] = A.grad.float().contiguous()
    res[f"gB.{l}.{name}"] = B.grad.float().contiguous()
save_file(res, out)
print("tokens", n, "targets", ntg, "sites", len(pairs), "loss", float(loss))
