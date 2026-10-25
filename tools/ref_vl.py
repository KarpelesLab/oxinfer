#!/usr/bin/env python3
"""Vision-language reference (Qwen3.5 / 3.6 / 3.8 images and videos, Gemma 4 videos): runs the HF
processor and model on the same inputs as oxinfer and compares with its dumps.

usage: ref_vl.py MODEL_DIR [--image F]... [--video FRAMES_DIR]... [--frames-fps 24]
                 [--video-fps F] [--video-frames N] [--prompt TEXT] [--pil] [--dtype f32|bf16]
                 [--ours PREFIX] [--ours-logits FILE] [--ours-prompt FILE] [--generate N]

Videos are directories of frames (sorted image files, e.g. `ffmpeg -i clip.mp4 dir/%05d.png`),
passed to the processor as an array with their frame rate, so it samples frames itself.
--pil uses the PIL image processor backend (bit-exact resize; videos always use torchvision).
--ours PREFIX compares oxinfer's --dump-mm files (PREFIX_pixels / _image / _vpixels / _video).
--ours-logits compares the last-position logits (oxinfer --dump-logits, no system prompt).
--ours-prompt compares the expanded prompt text (oxinfer --print-prompt --no-system).
"""
import argparse, os, sys, time
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--image", action="append", default=[])
ap.add_argument("--video", action="append", default=[])
ap.add_argument("--frames-fps", type=float, default=24.0)
ap.add_argument("--video-fps", type=float)
ap.add_argument("--video-frames", type=int)
ap.add_argument("--prompt", default="Describe this image.")
ap.add_argument("--pil", action="store_true")
ap.add_argument("--dtype", default="f32")
ap.add_argument("--ours")
ap.add_argument("--ours-logits")
ap.add_argument("--ours-prompt")
ap.add_argument("--generate", type=int, default=0)
a = ap.parse_args()

from PIL import Image
from transformers import AutoProcessor, AutoImageProcessor, AutoModelForImageTextToText, AutoConfig
from transformers.video_utils import VideoMetadata

cfg = AutoConfig.from_pretrained(a.model)
gemma = cfg.model_type.startswith("gemma")
kw = {"image_processor": AutoImageProcessor.from_pretrained(a.model, backend="pil")} if a.pil else {}
proc = AutoProcessor.from_pretrained(a.model, **kw)
print("image processor:", type(proc.image_processor).__name__, "| video processor:", type(getattr(proc, "video_processor", None)).__name__)

content, call = [], {}
if a.image:
    content += [{"type": "image"} for _ in a.image]
    call["images"] = [Image.open(p).convert("RGB") for p in a.image]
if a.video:
    vids, metas = [], []
    for d in a.video:
        files = sorted(f for f in os.listdir(d) if f.lower().endswith((".png", ".jpg", ".jpeg")))
        frames = np.stack([np.asarray(Image.open(os.path.join(d, f)).convert("RGB")) for f in files])
        vids.append(frames)
        metas.append(VideoMetadata(total_num_frames=len(files), fps=a.frames_fps, width=frames.shape[2], height=frames.shape[1],
                                   duration=len(files) / a.frames_fps))
    content += [{"type": "video"} for _ in a.video]
    call["videos"] = vids
    call["video_metadata"] = metas
    call["do_sample_frames"] = True
    if a.video_fps is not None:
        call["fps"] = a.video_fps
    if a.video_frames is not None:
        if gemma and a.video_fps is None:
            call["num_frames"] = a.video_frames
        else:
            call["max_frames"] = a.video_frames
content.append({"type": "text", "text": a.prompt})
messages = [{"role": "user", "content": content}]
tkw = {} if gemma else {"enable_thinking": False}
text = proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, **tkw)
inputs = proc(text=text, return_tensors="pt", **call)
ids = inputs["input_ids"][0].tolist()
expanded = proc.tokenizer.decode(ids)
print("prompt tokens:", len(ids))
if a.ours_prompt:
    ours = open(a.ours_prompt).read()
    expanded_cmp = expanded
    print("expanded prompt identical:", ours == expanded_cmp)
    if ours != expanded_cmp:
        i = next((i for i, (x, y) in enumerate(zip(ours, expanded_cmp)) if x != y), min(len(ours), len(expanded_cmp)))
        print("  first difference at", i, "ours", repr(ours[max(0, i - 40):i + 60]), "\n  ref ", repr(expanded_cmp[max(0, i - 40):i + 60]))
for k, v in inputs.items():
    if hasattr(v, "shape"):
        print(f"  {k}: {tuple(v.shape)} {v.dtype}" + (f" {v.tolist()}" if "grid" in k or "num_" in k else ""))

dtype = torch.bfloat16 if a.dtype == "bf16" else torch.float32
t0 = time.time()
model = AutoModelForImageTextToText.from_pretrained(a.model, dtype=dtype).eval()
print(f"loaded {type(model).__name__} in {time.time() - t0:.1f}s")
if gemma and "num_frames_per_video" in inputs:
    # transformers 5.20's Gemma4Model.forward does not pass num_frames_per_video on.
    _gvf = model.model.get_video_features
    _nf = inputs["num_frames_per_video"]

    def _get_video_features(pixel_values_videos, video_position_ids=None, num_frames_per_video=None, **kwargs):
        nf = _nf if num_frames_per_video is None else num_frames_per_video
        return _gvf(pixel_values_videos, video_position_ids, nf, **kwargs)

    model.model.get_video_features = _get_video_features


def load(path, cols):
    if not os.path.exists(path):
        return None
    return np.fromfile(path, dtype="<f4").reshape(-1, cols)


def report(name, ours, ref):
    if ours is None:
        return
    n = min(len(ours), len(ref))
    if len(ours) != len(ref):
        print(f"{name}: rows ours {len(ours)} vs ref {len(ref)}")
    o, r = ours[:n], ref[:n]
    d = np.abs(o - r)
    rel = np.linalg.norm(o - r) / max(np.linalg.norm(r), 1e-9)
    print(f"{name}: {n}x{r.shape[1]} max abs {d.max():.6f} mean abs {d.mean():.7f} rel {rel:.2e} (ref mean |x| {np.abs(r).mean():.4f})")


with torch.no_grad():
    mm = model.model
    if "pixel_values" in inputs:
        pv = inputs["pixel_values"]
        if gemma:
            f = mm.get_image_features(pv, inputs["image_position_ids"])
        else:
            f = mm.get_image_features(pv, inputs["image_grid_thw"])
        emb = torch.cat(list(f.pooler_output), 0).float().numpy()
        print("image tokens:", emb.shape)
        if a.ours:
            report("pixels", load(a.ours + "_pixels.bin", pv.shape[-1]), pv.reshape(-1, pv.shape[-1]).float().numpy())
            report("image features", load(a.ours + "_image.bin", emb.shape[1]), emb)
    if "pixel_values_videos" in inputs:
        pv = inputs["pixel_values_videos"]
        if gemma:
            f = mm.get_video_features(pv, inputs["video_position_ids"], inputs["num_frames_per_video"])
            valid = (inputs["video_position_ids"][0] != -1).all(-1)
            first = pv[0][valid].float().numpy()
        else:
            f = mm.get_video_features(pv, inputs["video_grid_thw"])
            first = pv.float().numpy()
        emb = torch.cat(list(f.pooler_output), 0).float().numpy()
        print("video tokens:", emb.shape)
        if a.ours:
            report("video pixels" + (" (frame 0)" if gemma else ""), load(a.ours + "_vpixels.bin", first.shape[-1]), first)
            report("video features", load(a.ours + "_video.bin", emb.shape[1]), emb)
    t0 = time.time()
    out = model(**inputs)
    ref = out.logits[0, -1].float().numpy()
    print(f"forward {time.time() - t0:.1f}s")
    rt = np.argsort(-ref)[:10]
    print("ref  top10:", [(int(i), round(float(ref[i]), 3)) for i in rt])
    if a.ours_logits:
        ours = np.fromfile(a.ours_logits, dtype="<f4")
        d = np.abs(ref - ours)
        ot = np.argsort(-ours)[:10]
        print("ours top10:", [(int(i), round(float(ours[i]), 3)) for i in ot])
        p = np.exp(ref - ref.max()); p /= p.sum()
        q = np.exp(ours - ours.max()); q /= q.sum()
        kl = float((p * (np.log(p + 1e-30) - np.log(q + 1e-30))).sum())
        print(f"logits: max abs diff {d.max():.5f} mean {d.mean():.6f} | KL(ref||ours) {kl:.2e} | argmax match {int(rt[0]) == int(ot[0])} | top10 same set {set(rt) == set(ot)}")
    if a.generate:
        g = model.generate(**inputs, max_new_tokens=a.generate, do_sample=False)
        new = g[0, inputs["input_ids"].shape[1]:].tolist()
        print("ref greedy ids:", new)
        print("ref greedy text:", repr(proc.tokenizer.decode(new)))
