#!/usr/bin/env python3
"""MiniMax M3 image input against transformers: a tiny random MiniMaxM3SparseForConditionalGeneration
(vision tower, projector, sparse MoE text model) with MiniMax-M3.1-Flash's tokenizer.

usage: minimax_vl_tiny.py build OUT_DIR TOKENIZER_DIR
         writes OUT_DIR (config.json, model.safetensors, tokenizer files)
       minimax_vl_tiny.py check OUT_DIR IMAGE PROMPT_FILE OURS_PREFIX OURS_LOGITS
         PROMPT_FILE: the prompt oxinfer formatted (`--print-prompt`, its image placeholder
         expanded); OURS_PREFIX: oxinfer's `--dump-mm` prefix; OURS_LOGITS: `--dump-logits`.
         Compares the pixels, the image embeddings and the last position's logits.
"""
import json, os, shutil, sys

import numpy as np
import torch

from transformers.models.minimax_m3_vl.configuration_minimax_m3_vl import (
    MiniMaxM3VLConfig, MiniMaxM3VLTextConfig, MiniMaxM3VLVisionConfig)
from transformers.models.minimax_m3_vl.modeling_minimax_m3_vl import MiniMaxM3SparseForConditionalGeneration

mode, out = sys.argv[1], sys.argv[2]
V = 200064

if mode == "build":
    tok_dir = sys.argv[3]
    torch.manual_seed(0)
    n = 3
    tc = MiniMaxM3VLTextConfig(
        vocab_size=V, hidden_size=128, intermediate_size=64, dense_intermediate_size=192,
        shared_intermediate_size=64, num_hidden_layers=n, num_attention_heads=4, num_key_value_heads=2,
        head_dim=64, num_local_experts=8, num_experts_per_tok=2, rotary_dim=32,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.5},
        mlp_layer_types=["dense"] + ["sparse"] * (n - 1),
        layer_types=["full_attention"] + ["minimax_m3_sparse"] * (n - 1),
        index_n_heads=2, index_head_dim=64, index_block_size=16, index_topk_blocks=3, index_local_blocks=1,
        eos_token_id=200020, bos_token_id=200034, initializer_range=0.08,
    )
    # Heads of 80 as the release's (1280 / 16): 13 RoPE frequencies per axis, 2 pass-through dims.
    vc = MiniMaxM3VLVisionConfig(hidden_size=160, intermediate_size=256, num_hidden_layers=2,
                                 num_attention_heads=2, patch_size=14, spatial_merge_size=2,
                                 temporal_patch_size=2, initializer_range=0.08)
    cfg = MiniMaxM3VLConfig(text_config=tc.to_dict(), vision_config=vc.to_dict(), projector_hidden_size=96)
    cfg.text_config._attn_implementation = "eager"
    cfg.vision_config._attn_implementation = "eager"
    m = MiniMaxM3SparseForConditionalGeneration(cfg).float().eval()
    with torch.no_grad():
        for name, p in m.named_parameters():
            if "norm" in name and "vision" not in name and "layer_norm" not in name and "layrnorm" not in name:
                p.copy_(torch.randn_like(p) * 0.2)
            elif "layer_norm" in name or "layrnorm" in name:
                p.copy_((1.0 if name.endswith("weight") else 0.0) + torch.randn_like(p) * 0.1)
            elif name.endswith(".bias"):
                p.copy_(torch.randn_like(p) * 0.05)
        for name, b in m.named_buffers():
            if name.endswith("e_score_correction_bias"):
                b.copy_(torch.randn_like(b) * 0.3)
        for name, p in m.named_parameters():
            p.copy_(p.to(torch.bfloat16).float())
    m.save_pretrained(out)
    for f in ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "special_tokens_map.json", "added_tokens.json"]:
        if os.path.exists(os.path.join(tok_dir, f)):
            shutil.copy(os.path.join(tok_dir, f), out)
    print("saved", out)
    sys.exit(0)

image, prompt_file, ours_prefix, ours_logits = sys.argv[3:7]
from PIL import Image
from transformers import AutoTokenizer
from transformers.models.minimax_m3_vl.image_processing_minimax_m3_vl import MiniMaxM3VLImageProcessor

m = MiniMaxM3SparseForConditionalGeneration.from_pretrained(out).float().eval()
tok = AutoTokenizer.from_pretrained(out)
ip = MiniMaxM3VLImageProcessor()
px = ip(images=[Image.open(image).convert("RGB")], return_tensors="pt")
pv, grid = px["pixel_values"].float(), px["image_grid_thw"]
print("grid", grid.tolist(), "pixel rows", pv.shape)


def cmp(name, a, b):
    a, b = np.asarray(a, np.float32).ravel(), np.asarray(b, np.float32).ravel()
    if a.size != b.size:
        print(f"{name}: size ours {a.size} vs ref {b.size}")
        return
    d = np.abs(a - b)
    print(f"{name}: max abs {d.max():.3e}, mean abs {d.mean():.3e}, ref scale {np.abs(b).mean():.3e}")


cmp("pixels", np.fromfile(f"{ours_prefix}_pixels.bin", "<f4"), pv.numpy())
with torch.no_grad():
    feats = m.model.get_image_features(pv, grid).pooler_output[0]
cmp("image embeddings", np.fromfile(f"{ours_prefix}_image.bin", "<f4"), feats.numpy())
text = open(prompt_file).read()
ids = tok(text, add_special_tokens=False, return_tensors="pt").input_ids
n_img = int((ids == m.config.image_token_id).sum())
print("prompt tokens", ids.shape[1], "image tokens", n_img, "features", feats.shape[0])
with torch.no_grad():
    logits = m(input_ids=ids, pixel_values=pv, image_grid_thw=grid).logits[0, -1]
ours = np.fromfile(ours_logits, "<f4")
cmp("last logits", ours, logits.numpy())
print("argmax ours", int(ours.argmax()), "ref", int(logits.argmax()))
