#!/usr/bin/env python3
"""Janus-Pro reference dumps (the official `janus` package, float32, CPU), in the formats of
`cargo run --release --example janus_check` (same modes, same file names):

  python tools/ref_janus.py MODEL text QUESTION OUTDIR          prompt ids + last-token text logits
  python tools/ref_janus.py MODEL gen PROMPT STEPS OUTDIR       greedy CFG image tokens + guided logits
  python tools/ref_janus.py MODEL vq TOKENS.txt OUT.bin         VQ decoder output (HWC f32 in [-1, 1])
  python tools/ref_janus.py MODEL vision IMAGE QUESTION OUTDIR  pixels, features, embeddings, logits,
                                                                greedy answer (MAXTOK tokens, 64)

Needs torch, torchvision, transformers 4.4x, timm, einops, pillow and a checkout of
github.com/deepseek-ai/Janus in $JANUS_SRC. attrdict (imported by janus) does not run on
Python >= 3.10; a minimal stand-in is installed below when it is missing.
"""
import os
import sys
import types

sys.path.insert(0, os.environ.get("JANUS_SRC", "Janus"))
try:
    import attrdict  # noqa: F401
except ImportError:

    class AttrDict(dict):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            for key, v in list(self.items()):
                if isinstance(v, dict) and not isinstance(v, AttrDict):
                    self[key] = AttrDict(v)

        def __getattr__(self, k):
            try:
                return self[k]
            except KeyError:
                raise AttributeError(k)

        def __setattr__(self, k, v):
            self[k] = v

    sys.modules["attrdict"] = types.SimpleNamespace(AttrDict=AttrDict)

import numpy as np
import torch
from janus.models import MultiModalityCausalLM, VLChatProcessor
from PIL import Image

torch.set_grad_enabled(False)
model_path, mode = sys.argv[1], sys.argv[2]
proc = VLChatProcessor.from_pretrained(model_path)
tok = proc.tokenizer
m = MultiModalityCausalLM.from_pretrained(model_path, torch_dtype=torch.float32).eval()


def save(path, a):
    np.ascontiguousarray(a, dtype=np.float32).tofile(path)


if mode == "text":
    q, out = sys.argv[3], sys.argv[4]
    os.makedirs(out, exist_ok=True)
    conv = [{"role": "<|User|>", "content": q}, {"role": "<|Assistant|>", "content": ""}]
    sft = proc.apply_sft_template_for_multi_turn_prompts(conv, sft_format=proc.sft_format, system_prompt=proc.system_prompt)
    ids = tok.encode(sft)
    open(f"{out}/ids.txt", "w").write(" ".join(map(str, ids)))
    o = m.language_model(input_ids=torch.tensor([ids]))
    save(f"{out}/logits.bin", o.logits[0, -1].numpy())
    print(repr(sft), len(ids))
elif mode == "gen":
    prompt, steps, out = sys.argv[3], int(sys.argv[4]), sys.argv[5]
    os.makedirs(out, exist_ok=True)
    conv = [{"role": "<|User|>", "content": prompt}, {"role": "<|Assistant|>", "content": ""}]
    sft = proc.apply_sft_template_for_multi_turn_prompts(conv, sft_format=proc.sft_format, system_prompt="")
    sft = sft + proc.image_start_tag
    ids = torch.LongTensor(tok.encode(sft))
    open(f"{out}/ids.txt", "w").write(" ".join(map(str, ids.tolist())))
    tokens = torch.zeros((2, len(ids)), dtype=torch.int)
    tokens[0] = ids
    tokens[1] = ids
    tokens[1, 1:-1] = proc.pad_id
    emb = m.language_model.get_input_embeddings()(tokens)
    gen = []
    logs = []
    past = None
    for i in range(steps):
        o = m.language_model.model(inputs_embeds=emb, use_cache=True, past_key_values=past)
        past = o.past_key_values
        lg = m.gen_head(o.last_hidden_state[:, -1, :])
        g = lg[1] + 5.0 * (lg[0] - lg[1])
        logs.append(g.numpy())
        nt = int(torch.argmax(g))
        gen.append(nt)
        nt2 = torch.tensor([nt, nt])
        emb = m.prepare_gen_img_embeds(nt2).unsqueeze(1)
    open(f"{out}/tokens.txt", "w").write(" ".join(map(str, gen)))
    save(f"{out}/guided.bin", np.stack(logs))
    print(repr(sft), len(ids), gen[:16])
elif mode == "vq":
    toks = [int(t) for t in open(sys.argv[3]).read().split()]
    out = sys.argv[4]
    dec = m.gen_vision_model.decode_code(torch.tensor([toks], dtype=torch.int), shape=[1, 8, 24, 24])
    a = dec[0].permute(1, 2, 0).numpy()
    save(out, a)
    img = np.clip((a + 1) / 2 * 255, 0, 255)
    v = np.zeros((384, 384, 3), dtype=np.uint8)
    v[:, :] = img
    Image.fromarray(v).save(out + ".png")
    print(a.shape, a.min(), a.max())
elif mode == "vision":
    path, q, out = sys.argv[3], sys.argv[4], sys.argv[5]
    os.makedirs(out, exist_ok=True)
    conv = [
        {"role": "<|User|>", "content": f"<image_placeholder>\n{q}", "images": [path]},
        {"role": "<|Assistant|>", "content": ""},
    ]
    img = Image.open(path).convert("RGB")
    inp = proc(conversations=conv, images=[img], force_batchify=True)
    pv = inp.pixel_values.float()
    save(f"{out}/pixels.bin", pv[0, 0].numpy())
    feats = m.vision_model(pv[0])
    save(f"{out}/features.bin", feats[0].numpy())
    save(f"{out}/embeds.bin", m.aligner(feats)[0].numpy())
    ids = inp.input_ids[0].tolist()
    open(f"{out}/ids.txt", "w").write(" ".join(map(str, ids)))
    e = m.prepare_inputs_embeds(**{k: inp[k] for k in ["input_ids", "pixel_values", "images_seq_mask", "images_emb_mask"]})
    o = m.language_model(inputs_embeds=e, attention_mask=inp.attention_mask)
    save(f"{out}/logits.bin", o.logits[0, -1].numpy())
    g = m.language_model.generate(
        inputs_embeds=e,
        attention_mask=inp.attention_mask,
        pad_token_id=tok.eos_token_id,
        bos_token_id=tok.bos_token_id,
        eos_token_id=tok.eos_token_id,
        max_new_tokens=int(os.environ.get("MAXTOK", "64")),
        do_sample=False,
        use_cache=True,
    )
    ans = tok.decode(g[0].tolist(), skip_special_tokens=True)
    open(f"{out}/answer.txt", "w").write(ans)
    print(repr(inp.sft_format[0]), len(ids))
    print(ans)
