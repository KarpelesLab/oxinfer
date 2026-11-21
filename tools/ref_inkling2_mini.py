"""transformers reference outputs of a real Inkling-2 loop checkpoint (thinkingmachines/Inkling-2-Mini),
computed in float32 on the weights exactly as stored.

usage: ref_inkling2_mini.py MODEL_DIR OUT.safetensors [--chat CONVERSATION.json | --text TEXT]
                            [--gen N] [--think] [--hidden HIDDEN.safetensors]

transformers dequantizes the release's NVFP4 / MXFP8 experts on load into the model's dtype. Here
they are kept in bfloat16 without their per-expert factor (`amax / outer_scale_denominator`):
E2M1 code times E4M3 scale, and E4M3 value times a power of two, are exact in bfloat16. The
experts then run in float32 with the factor applied to the upcast weights, and every other
parameter is float32: the reference is the stored model in float32 (about 56 GB of memory).

Writes `ids` (I64), `logits` [len, vocab] (one uncached forward), `gen` (I64, greedy tokens) and
`decode_logits` [gen, vocab] (after feeding each through the cache), the format
`examples/inkling2_check.rs` reads.
"""
import json, os, sys
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

import transformers.core_model_loading as cml
from transformers import AutoTokenizer
from transformers.models.inkling2_loop import modeling_inkling2_loop as mod

model_dir, out = sys.argv[1], sys.argv[2]
a = sys.argv[3:]
chat, text, gen, think, hidden = None, None, 16, False, None
for i, x in enumerate(a):
    if x == "--chat":
        chat = a[i + 1]
    if x == "--text":
        text = a[i + 1]
    if x == "--gen":
        gen = int(a[i + 1])
    if x == "--think":
        think = True
    if x == "--hidden":
        hidden = a[i + 1]

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
        final.index_add_(0, tok, hcur)
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
    # The factors from the files: the loader hands the conversion `amax` already cast to the
    # model's dtype (bfloat16), which would change the experts by up to 0.4%.
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
                if k + "_amax" in raw:
                    fac = raw[k + "_amax"] / raw[k + "_outer_scale_denominator"]
                else:
                    fac = torch.ones(mm.num_experts)
                setattr(mm, short + "_factor", fac)

tk = AutoTokenizer.from_pretrained(model_dir)
if chat:
    conv = json.load(open(chat))
    more = {k: v for k, v in conv.items() if k not in ("messages", "tools")}
    prompt = tk.apply_chat_template(conv["messages"], tools=conv.get("tools"), add_generation_prompt=True, tokenize=False, **more)
else:
    msgs = [{"role": "user", "content": text or "Why is the sky blue? Answer in two sentences."}]
    prompt = tk.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, reasoning_effort=0.9 if think else 0.0)
ids = torch.tensor([tk(prompt, add_special_tokens=False)["input_ids"]])
print(repr(prompt))
L = "logits"
with torch.no_grad():
    o = m(ids, use_cache=False, output_hidden_states=hidden is not None)
    full = o[L][0].float()
    if hidden:
        # The embeddings, then the output of every layer application (the loop unrolled).
        save_file({"hidden": torch.stack([h[0].float() for h in o.hidden_states]).contiguous()}, hidden)
    o = m(ids, use_cache=True)
    cache, last = o.cache, o[L][0, -1].float()
    toks, rows = [], []
    for _ in range(gen):
        t = int(last.argmax())
        toks.append(t)
        o = m(torch.tensor([[t]]), cache=cache, use_cache=True)
        cache, last = o.cache, o[L][0, -1].float()
        rows.append(last)
save_file(
    {
        "ids": ids[0].to(torch.int64),
        L: full.contiguous(),
        "gen": torch.tensor(toks, dtype=torch.int64),
        "decode_" + L: torch.stack(rows).contiguous(),
    },
    out,
)
print("prompt", ids.shape[1], "tokens; greedy:", repr(tk.decode(toks)))
