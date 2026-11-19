#!/usr/bin/env python3
"""MiniMax-H3 `ref2va`'s packed layout (diffusers' `MiniMaxH3Ref2VAPrepareLayoutStep.
build_ref2va_packed_sequence`) on a fixed request, for examples/h3_ref2va_check.rs.

usage: minimax_h3_ref2va_layout.py OUT_DIR

The request: 11 text rows (rows 3 to 6 a vision block, tagged video), then an image reference
(one latent frame of 8 x 12), a video reference with its soundtrack (7 latent frames of 6 x 10,
207 audio latents), an audio reference (57 latents) and a second image (10 x 6); a target of 12
latent frames of 6 x 10 with 207 audio latents, patch 2. Writes OUT_DIR/ref_layout.json:
the references' geometry and the expected positions (float64), tags and row indices.
"""
import json, os, sys

import torch
from diffusers.modular_pipelines.minimax_h3.before_denoise import MiniMaxH3Ref2VAPrepareLayoutStep


class Ref:
    def __init__(self, kind, has_audio):
        self.kind, self.has_audio = kind, has_audio


out = sys.argv[1]
os.makedirs(out, exist_ok=True)
text_tags = torch.tensor([1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1])
refs = [Ref("image", False), Ref("video", True), Ref("audio", True), Ref("image", False)]
cond = [torch.zeros(1, 4, 1, 8, 12), torch.zeros(1, 4, 7, 6, 10), torch.zeros(1, 4, 1, 10, 6)]
audio = [torch.zeros(207 * 2, 6), torch.zeros(57 * 2, 6)]
lt, lh, lw, na = 12, 6, 10, 207
pos, tags, vidx, aidx, tidx, ncv, nca = MiniMaxH3Ref2VAPrepareLayoutStep.build_ref2va_packed_sequence(
    text_tags, refs, cond, audio, lt, lh, lw, na, (1, 2, 2), 2, 2, 0
)
json.dump({
    "text_tags": text_tags.tolist(),
    "refs": [{"kind": r.kind, "visual": None, "audio": 0} for r in refs],
    "visual": [list(c.shape[2:]) for c in cond],
    "audio": [a.shape[0] // 2 for a in audio],
    "target": [lt, lh, lw, na],
    "positions": pos.tolist(), "tags": tags.tolist(), "video_index": vidx.tolist(),
    "audio_index": aidx.tolist(), "n_cond": ncv, "n_cond_audio": nca,
}, open(os.path.join(out, "ref_layout.json"), "w"))
print(f"{pos.shape[0]} rows, {ncv} reference video rows, {nca} reference audio rows")
