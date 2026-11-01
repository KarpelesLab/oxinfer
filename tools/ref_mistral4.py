#!/usr/bin/env python3
# Reference outputs of a (tiny) Mistral Large 4 checkpoint with transformers in float32:
#   prefill: logits at every position, hidden states after each layer, the DSA top-k of every
#   full indexer layer; decode: greedy steps with the KV cache, logits at each step.
# usage: ref_mistral4.py MODEL_DIR OUT.safetensors N_PROMPT N_DECODE [SEED] [--weights state.pt]
import sys

import torch
from safetensors.torch import save_file

from transformers import MistralLarge4ForConditionalGeneration

d, out, n, nd = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
seed = int(sys.argv[5]) if len(sys.argv) > 5 and not sys.argv[5].startswith("--") else 0
weights = sys.argv[sys.argv.index("--weights") + 1] if "--weights" in sys.argv else None
torch.set_num_threads(8)
model = MistralLarge4ForConditionalGeneration.from_pretrained(d, dtype=torch.float32, attn_implementation="eager").eval()
if weights:
    missing = model.load_state_dict(torch.load(weights), strict=False)
    print("loaded", weights, missing)
g = torch.Generator().manual_seed(seed)
ids = torch.cat([torch.tensor([1]), torch.randint(1000, 204800, (n - 1,), generator=g)]).unsqueeze(0)

topk = {}
lm = model.model.language_model
for i, layer in enumerate(lm.layers):
    ix = layer.self_attn.indexer
    if ix is not None:
        def hook(mod, inp, outp, i=i):
            topk.setdefault(i, []).append(outp.detach().clone())
        ix.register_forward_hook(hook)

with torch.no_grad():
    o = model(input_ids=ids, output_hidden_states=True, use_cache=False)
res = {"ids": ids[0].to(torch.int64), "logits": o.logits[0].float()}
for i, h in enumerate(o.hidden_states):
    res[f"hidden.{i}"] = h[0].float()
for i, v in topk.items():
    t = v[0][0].to(torch.int64)
    res[f"topk.{i}"] = torch.sort(t, dim=-1).values
topk.clear()

# Greedy decoding with the cache.
with torch.no_grad():
    o = model(input_ids=ids, use_cache=True)
    past = o.past_key_values
    nxt = o.logits[0, -1].argmax()
    gen, steps = [], []
    for s in range(nd):
        gen.append(int(nxt))
        o = model(input_ids=nxt.view(1, 1), past_key_values=past, use_cache=True)
        past = o.past_key_values
        steps.append(o.logits[0, -1].float())
        nxt = o.logits[0, -1].argmax()
res["gen"] = torch.tensor(gen, dtype=torch.int64)
if steps:
    res["decode_logits"] = torch.stack(steps)
save_file({k: v.contiguous() for k, v in res.items()}, out)
print("prompt", ids.shape[1], "gen", gen)
print("top5 last", torch.topk(res["logits"][-1], 5))
