#!/usr/bin/env python3
"""Janus-Pro and JanusFlow reference dumps (the official `janus` package, float32, CPU), in the
formats of `cargo run --release --example janus_check` (same modes, same file names); the variant
is read from the checkpoint's config:

  python tools/ref_janus.py MODEL text QUESTION OUTDIR          prompt ids + last-token text logits
  python tools/ref_janus.py MODEL gen PROMPT STEPS OUTDIR       greedy CFG image tokens + guided logits
  python tools/ref_janus.py MODEL vq TOKENS.txt OUT.bin         VQ decoder output (HWC f32 in [-1, 1])
  python tools/ref_janus.py MODEL vision IMAGE QUESTION OUTDIR  pixels, features, embeddings, logits,
                                                                greedy answer (MAXTOK tokens, 64)
  python tools/ref_janus.py MODEL chat IMAGE Q1 ANSWER1.txt Q2 OUTDIR
                                                                two turns (the answer to Q1 given):
                                                                ids + logits after Q2
  python tools/ref_janus.py MODEL flow PROMPT NOISE.bin STEPS CFG OUTDIR
                                                                JanusFlow from the noise in NOISE.bin
                                                                ([48, 48, 4] f32, written with seed 0
                                                                when missing): velocity.bin (per step,
                                                                conditional then unconditional, HWC),
                                                                latents.bin, pixels.bin, image.png.
                                                                $JANUS_VAE: the SDXL VAE (default
                                                                stabilityai/sdxl-vae)

Needs torch, torchvision, transformers 4.4x, timm, einops, pillow (JanusFlow: diffusers too) and a
checkout of github.com/deepseek-ai/Janus in $JANUS_SRC. attrdict (imported by janus) does not run on
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

import json

import numpy as np
import torch
from PIL import Image

torch.set_grad_enabled(False)
model_path, mode = sys.argv[1], sys.argv[2]
FLOW = "vision_gen_enc_config" in json.load(open(os.path.join(model_path, "config.json")))
if FLOW:
    from janus.janusflow.models import MultiModalityCausalLM, VLChatProcessor

    USER, ASSISTANT = "User", "Assistant"
else:
    from janus.models import MultiModalityCausalLM, VLChatProcessor

    USER, ASSISTANT = "<|User|>", "<|Assistant|>"
proc = VLChatProcessor.from_pretrained(model_path)
tok = proc.tokenizer
m = MultiModalityCausalLM.from_pretrained(model_path, torch_dtype=torch.float32).eval()
vision_model = m.vision_und_enc_model if FLOW else m.vision_model
aligner = m.vision_und_enc_aligner if FLOW else m.aligner


def save(path, a):
    np.ascontiguousarray(a, dtype=np.float32).tofile(path)


if mode == "text":
    q, out = sys.argv[3], sys.argv[4]
    os.makedirs(out, exist_ok=True)
    conv = [{"role": USER, "content": q}, {"role": ASSISTANT, "content": ""}]
    sft = proc.apply_sft_template_for_multi_turn_prompts(conv, sft_format=proc.sft_format, system_prompt=proc.system_prompt)
    ids = tok.encode(sft)
    open(f"{out}/ids.txt", "w").write(" ".join(map(str, ids)))
    o = m.language_model(input_ids=torch.tensor([ids]))
    save(f"{out}/logits.bin", o.logits[0, -1].numpy())
    print(repr(sft), len(ids))
elif mode == "gen":
    prompt, steps, out = sys.argv[3], int(sys.argv[4]), sys.argv[5]
    os.makedirs(out, exist_ok=True)
    conv = [{"role": USER, "content": prompt}, {"role": ASSISTANT, "content": ""}]
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
        {"role": USER, "content": f"<image_placeholder>\n{q}", "images": [path]},
        {"role": ASSISTANT, "content": ""},
    ]
    img = Image.open(path).convert("RGB")
    inp = proc(conversations=conv, images=[img], force_batchify=True)
    pv = inp.pixel_values.float()
    save(f"{out}/pixels.bin", pv[0, 0].numpy())
    feats = vision_model(pv[0])
    save(f"{out}/features.bin", feats[0].numpy())
    save(f"{out}/embeds.bin", aligner(feats)[0].numpy())
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
elif mode == "chat":
    path, q1, a1, q2, out = sys.argv[3], sys.argv[4], open(sys.argv[5]).read(), sys.argv[6], sys.argv[7]
    os.makedirs(out, exist_ok=True)
    conv = [
        {"role": USER, "content": f"<image_placeholder>\n{q1}", "images": [path]},
        {"role": ASSISTANT, "content": a1},
        {"role": USER, "content": q2},
        {"role": ASSISTANT, "content": ""},
    ]
    img = Image.open(path).convert("RGB")
    inp = proc(conversations=conv, images=[img], force_batchify=True)
    ids = inp.input_ids[0].tolist()
    open(f"{out}/ids.txt", "w").write(" ".join(map(str, ids)))
    e = m.prepare_inputs_embeds(**{k: inp[k] for k in ["input_ids", "pixel_values", "images_seq_mask", "images_emb_mask"]})
    o = m.language_model(inputs_embeds=e, attention_mask=inp.attention_mask)
    save(f"{out}/logits.bin", o.logits[0, -1].numpy())
    print(repr(inp.sft_format[0]), len(ids))
elif mode == "flow":
    # The README's generate() for one image, in float32 (the prompt is run again at every step
    # there too: its past_key_values stay empty).
    from diffusers.models import AutoencoderKL

    prompt, noise, steps, cfg, out = sys.argv[3], sys.argv[4], int(sys.argv[5]), float(sys.argv[6]), sys.argv[7]
    os.makedirs(out, exist_ok=True)
    vae = AutoencoderKL.from_pretrained(os.environ.get("JANUS_VAE", "stabilityai/sdxl-vae")).eval()
    conv = [{"role": USER, "content": prompt}, {"role": ASSISTANT, "content": ""}]
    sft = proc.apply_sft_template_for_multi_turn_prompts(conv, sft_format=proc.sft_format, system_prompt="")
    sft = sft + proc.image_gen_tag
    ids = torch.LongTensor(tok.encode(sft))
    open(f"{out}/ids.txt", "w").write(" ".join(map(str, ids.tolist())))
    tokens = torch.stack([ids] * 2)
    tokens[1:, 1:] = proc.pad_id
    emb = m.language_model.get_input_embeddings()(tokens)[:, :-1, :]
    if not os.path.exists(noise):
        torch.manual_seed(0)
        save(noise, torch.randn((48, 48, 4)).numpy())
    z = torch.from_numpy(np.fromfile(noise, dtype=np.float32).reshape(1, 48, 48, 4)).permute(0, 3, 1, 2).contiguous()
    dt = 1.0 / steps
    dt = torch.zeros_like(z) + dt
    mask = torch.ones((2, emb.shape[1] + 577))
    mask[1:, 1 : emb.shape[1]] = 0
    mask = mask.int()
    vel = []
    for step in range(steps):
        z_input = torch.cat([z, z], dim=0)
        t = step / steps * 1000.0
        t = torch.tensor([t] * z_input.shape[0]).to(dt)
        z_emb, t_emb, hs = m.vision_gen_enc_model(z_input, t)
        z_emb = z_emb.view(z_emb.shape[0], z_emb.shape[1], -1).permute(0, 2, 1)
        z_emb = m.vision_gen_enc_aligner(z_emb)
        llm_emb = torch.cat([emb, t_emb.unsqueeze(1), z_emb], dim=1)
        o = m.language_model.model(inputs_embeds=llm_emb, use_cache=True, attention_mask=mask, past_key_values=None)
        h = m.vision_gen_dec_aligner(m.vision_gen_dec_aligner_norm(o.last_hidden_state[:, -576:, :]))
        h = h.reshape(z_emb.shape[0], 24, 24, 768).permute(0, 3, 1, 2)
        v = m.vision_gen_dec_model(h, hs, t_emb)
        vel.append(v.permute(0, 2, 3, 1).numpy().copy())
        v_cond, v_uncond = torch.chunk(v, 2)
        v = cfg * v_cond - (cfg - 1.0) * v_uncond
        z = z + dt * v
        print(f"step {step + 1}/{steps}", flush=True)
    save(f"{out}/velocity.bin", np.stack(vel))
    save(f"{out}/latents.bin", z[0].permute(1, 2, 0).numpy())
    dec = vae.decode(z / vae.config.scaling_factor).sample
    save(f"{out}/pixels.bin", dec[0].permute(1, 2, 0).numpy())
    img = dec.float().clip_(-1.0, 1.0).permute(0, 2, 3, 1).numpy()
    Image.fromarray(((img[0] + 1) / 2.0 * 255).astype(np.uint8)).save(f"{out}/image.png")
    print(repr(sft), len(ids))
