#!/usr/bin/env python3
"""Reference outputs of baidu/Unlimited-OCR-2 from the checkpoint's own code, in float32 on the
CPU, compared with oxinfer's dumps.

    M=~/.cache/oxinfer/hub/baidu--Unlimited-OCR-2
    oxinfer --model $M --image page.png --device cpu --dump-mm ours --dump-logits ours.bin
    oxinfer --model $M --image page.png --device cpu --show-tokens -n 64 2> ours.log
    python tools/ref_unlimited_ocr.py $M --image page.png --ours ours --ours-logits ours.bin \
        --ours-log ours.log --generate 64

Several --image: one page each (base mode, as infer() chooses). The checks, in order: token ids
of the prompt (with --ours-log), pixel values, encoder features of the global view, the packed
image embeddings, the logits after the prompt, the greedy tokens.

Three things differ from running `model.infer()` as published, none of them in the math:
the model's `.cuda()` call on the image mask is made a no-op (CPU), the pixel tensors (which
`_uocr_build_inputs` rounds to bfloat16) are widened to float32 to match float32 weights, and
the attention runs the code's own dense fallback (`DashAttention._forward_dense`,
`_decode_attend`) since its Triton kernels need a GPU.
"""
import argparse
import json
import re
import sys

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer


def diff(name, ours, ref):
    ours, ref = np.asarray(ours, np.float32).ravel(), np.asarray(ref, np.float32).ravel()
    if ours.shape != ref.shape:
        print(f"{name}: SHAPE MISMATCH ours {ours.shape} ref {ref.shape}")
        return
    d = np.abs(ours - ref)
    cos = float(ours @ ref / (np.linalg.norm(ours) * np.linalg.norm(ref) + 1e-30))
    print(f"{name}: n={ref.size} max|diff|={d.max():.3e} mean|diff|={d.mean():.3e} max|ref|={np.abs(ref).max():.3e} cos={cos:.8f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--image", action="append", default=[])
    ap.add_argument("--prompt", default="")
    ap.add_argument("--ours", help="prefix of --dump-mm")
    ap.add_argument("--ours-logits")
    ap.add_argument("--ours-log", help="stderr of a --show-tokens run (prompt and output ids)")
    ap.add_argument("--generate", type=int, default=0)
    ap.add_argument("--ngram", type=int, default=0)
    ap.add_argument("--ngram-window", type=int, default=0)
    ap.add_argument("--dump", help="prefix to write the reference tensors to")
    a = ap.parse_args()

    torch.manual_seed(0)
    torch.Tensor.cuda = lambda self, *args, **kw: self
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(a.model, trust_remote_code=True, use_safetensors=True, torch_dtype=torch.float32).eval()
    mod = sys.modules[type(model).__module__]

    # float32 pixels (already rounded to bfloat16 by the builder).
    build = mod._uocr_build_inputs

    def build32(*args, **kw):
        ids, imgs, mask, spatial = build(*args, **kw)
        if imgs is not None:
            imgs = (imgs[0].float(), imgs[1].float())
        return ids, imgs, mask, spatial

    mod._uocr_build_inputs = build32

    images = [Image.open(p).convert("RGB") for p in a.image]
    n_img = len(images)
    # infer()'s prompt rules.
    prompt = a.prompt or ("Multi page parsing." if n_img else "")
    n_ph = prompt.count("<image>")
    if n_img and n_ph == 0:
        prompt = "<PAGE><image>" * n_img + prompt
    elif n_img > 1 and n_ph == 1:
        prompt = prompt.replace("<image>", "<PAGE><image>" * n_img)
    input_ids, imgs, seq_mask, spatial = build32(tok, prompt, images)
    ids = input_ids.tolist()
    print(f"prompt: {len(ids)} tokens, {int(seq_mask.sum())} image tokens, tiles {spatial.tolist() if spatial is not None else None}")

    ours_ids = ours_out = None
    if a.ours_log:
        log = open(a.ours_log).read()
        m = re.search(r"\[prompt ids \(\d+\): \[([^\]]*)\]", log)
        ours_ids = [int(x) for x in m.group(1).split(",")] if m else None
        m = re.search(r"\[output ids: \[([^\]]*)\]", log)
        ours_out = [int(x) for x in m.group(1).split(",") if x.strip()] if m else None
        if ours_ids is not None:
            print("prompt ids:", "identical" if ours_ids == ids else f"DIFFERENT (ours {len(ours_ids)}, ref {len(ids)})")

    if imgs is not None:
        crop, glob = imgs
        px = glob[0].numpy().ravel()
        if spatial[0].tolist() != [1, 1]:
            px = np.concatenate([px, crop[: int(spatial[0][0] * spatial[0][1])].numpy().ravel()])
        with torch.no_grad():
            f1 = model.model.sam_model(glob[:1])
            f2 = model.model.vision_model(glob[:1], f1)
            feats = torch.cat((f2[:, 1:], f1.flatten(2).permute(0, 2, 1)), dim=-1)[0].numpy()
        if a.dump:
            px.astype(np.float32).tofile(a.dump + "_pixels.bin")
            feats.astype(np.float32).tofile(a.dump + "_features.bin")
        if a.ours:
            diff("pixel values (first page: global view, tiles)", np.fromfile(a.ours + "_pixels.bin", np.float32), px)
            diff("encoder features (first page, global view)", np.fromfile(a.ours + "_features.bin", np.float32), feats)

    # The embeddings entering the decoder (image positions hold the packed features).
    captured = {}
    base_forward = mod.DeepseekV2Model.forward

    def spy(self, *args, **kw):
        if "embeds" not in captured and kw.get("inputs_embeds") is not None:
            captured["embeds"] = kw["inputs_embeds"].detach().clone()
        return base_forward(self, *args, **kw)

    mod.DeepseekV2Model.forward = spy

    # Prefill exactly as _uocr_generate does, for the logits after the prompt.
    from transformers.cache_utils import DynamicCache

    cfg = model.config
    saved = (getattr(cfg, "_ring_window", None), getattr(cfg, "sliding_window", None))
    cfg._ring_window = None
    cfg.sliding_window = None
    state = mod.configure_dash_inference(model.model, True, topm=16)
    try:
        kw = {}
        if imgs is not None:
            kw = dict(images=[(imgs[0], imgs[1], spatial)], images_seq_mask=seq_mask.unsqueeze(0), images_spatial_crop=spatial)
        with torch.no_grad():
            out = model.model(
                input_ids=input_ids.unsqueeze(0),
                past_key_values=DynamicCache(),
                use_cache=True,
                position_ids=torch.arange(len(ids))[None],
                return_dict=True,
                **kw,
            )
            logits = model.lm_head(out.last_hidden_state[:, -1:]).float()[0, -1].numpy()
    finally:
        mod.configure_dash_inference(model.model, False, state=state)
        cfg._ring_window, cfg.sliding_window = saved
    if imgs is not None:
        first = int(spatial[0][0] * spatial[0][1])
        n_first = 273 + ((10 * int(spatial[0][0]) + 1) * 10 * int(spatial[0][1]) if first > 1 else 0)
        emb = captured["embeds"][0][seq_mask][:n_first].numpy()
        if a.dump:
            emb.astype(np.float32).tofile(a.dump + "_image.bin")
        if a.ours:
            diff("image embeddings (first page, packed)", np.fromfile(a.ours + "_image.bin", np.float32), emb)
    top = np.argsort(-logits)[:10]
    print("ref top10:", " ".join(f"{i}:{logits[i]:.3f}" for i in top))
    if a.dump:
        logits.astype(np.float32).tofile(a.dump + "_logits.bin")
    if a.ours_logits:
        ours = np.fromfile(a.ours_logits, np.float32)
        diff("logits after the prompt", ours, logits)
        otop = np.argsort(-ours)[:10]
        print("ours top10:", " ".join(f"{i}:{ours[i]:.3f}" for i in otop))
        print("top10:", "same tokens, same order" if list(otop) == list(top) else "DIFFERENT")

    if a.generate > 0:
        res = model.infer(
            tok,
            prompt=a.prompt,
            image_file=images if n_img != 1 else images[0],
            eval_mode=True,
            max_new_tokens=a.generate,
            no_repeat_ngram_size=a.ngram,
            ngram_window=a.ngram_window,
            return_dict=True,
        )
        gen = res["token_ids"]
        print(f"ref greedy ({len(gen)} tokens, {res['finish_reason']}): {gen}")
        print(repr(res["text"]))
        if ours_out is not None:
            ours = [t for t in ours_out if t != tok.eos_token_id]
            n = min(len(ours), len(gen))
            same = next((i for i in range(n) if ours[i] != gen[i]), n)
            if same == len(gen) == len(ours):
                print(f"greedy tokens: identical ({same})")
            else:
                print(f"greedy tokens: first {same} identical (ours {len(ours)}, ref {len(gen)})")


if __name__ == "__main__":
    main()
