#!/usr/bin/env python3
"""Tiny random MiniMax-H3 (video) components (diffusers' AutoencoderKLMiniMaxH3 and
MiniMaxH3Transformer3DModel) and their outputs, for examples/h3_video_check.rs.

usage: minimax_h3_video_tiny.py OUT_DIR
Writes OUT_DIR/vae, OUT_DIR/transformer (save_pretrained), inputs and reference outputs (*.bin,
little-endian f32) and OUT_DIR/case.json. Then:
cargo run --release --example h3_video_check -- OUT_DIR

The VAE cases: a 22-frame clip and a single frame encoded (2 x 2 tiles), latents of 7 frames (one
clip), 12 (two clips, cross-faded) and 9 (padded) decoded. The DiT case: the packed
`[text | first and last keyframe rows | audio | video]` sequence of diffusers' layout step, the
first step's velocities and the rows after the whole schedule (video shift 12, audio shift 3,
keyframe rows held at t = 0.999).
"""
import json, os, sys

import numpy as np
import torch
from diffusers import AutoencoderKLMiniMaxH3, MiniMaxH3Scheduler, MiniMaxH3Transformer3DModel
from diffusers.modular_pipelines.minimax_h3.before_denoise import (
    MiniMaxH3PrepareLayoutStep,
    MiniMaxH3SetTimestepsStep,
    patchify_video_latents,
)

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
torch.manual_seed(0)


def randomize(m, scale=0.1):
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "norm" in n and n.endswith("weight"):
                p.copy_(1.0 + torch.randn_like(p) * 0.2)
            else:
                p.copy_(torch.randn_like(p) * scale)
            p.copy_(p.to(torch.bfloat16).float())


def save(name, t):
    np.asarray(t, dtype=np.float32).tofile(os.path.join(out, name))


case = {}

# ---- video VAE --------------------------------------------------------------------------------
C = 4
vae = AutoencoderKLMiniMaxH3(
    latent_channels=C, block_out_channels=(8, 16, 16, 16, 16, 32), norm_num_groups=4, decoder_num_layers=2,
    decoder_num_attention_heads=2, decoder_attention_head_dim=24, decoder_num_register_tokens=2,
    latents_mean=tuple(float(v) for v in torch.randn(C) * 0.3), latents_std=tuple(float(v) for v in torch.rand(C) + 0.5),
).float().eval()
randomize(vae, 0.05)
vae.save_pretrained(os.path.join(out, "vae"))
H, W = 288, 320  # 2 x 2 tiles of 256 overlapping by at least 64
video = torch.randn(1, 3, 22, H, W)
with torch.no_grad():
    m = vae.encode(video, return_dict=False)[0].parameters
    m1 = vae.encode(video[:, :, :1], return_dict=False)[0].parameters
save("video.bin", video[0].numpy())
save("moments.bin", m[0].numpy())
save("moments1.bin", m1[0].numpy())
lh, lw = H // 16, W // 16
mean = torch.tensor(vae.config.latents_mean).view(-1, 1, 1, 1)
std = torch.tensor(vae.config.latents_std).view(-1, 1, 1, 1)
dec = {}
for t in (7, 12, 9):
    lat = torch.randn(C, t, lh, lw)
    with torch.no_grad():
        px = vae.decode((lat * std + mean)[None], return_dict=False)[0][0]
    save(f"latents_{t}.bin", lat.numpy())
    save(f"decoded_{t}.bin", px.numpy())
    dec[str(t)] = px.shape[1]
case.update({"h": H, "w": W, "frames": 22, "moments": list(m.shape[1:]), "decode": dec})
print("moments", tuple(m.shape), tuple(m1.shape), "decoded frames", dec, "tiling", vae.use_tiling)

# ---- DiT --------------------------------------------------------------------------------------
dit = MiniMaxH3Transformer3DModel(
    num_attention_heads=2, attention_head_dim=32, hidden_size=48, num_layers=2, num_refiner_layers=1, ffn_dim=64,
    in_channels=4, audio_in_channels=6, patch_size=(1, 2, 2), text_dim=40, freq_dim=32, time_embed_hidden_dim=48,
    time_embed_dim=24, rope_freq_dim=4,
).float().eval()
randomize(dit)
dit.save_pretrained(os.path.join(out, "transformer"))
n_text, lt, dh, dw, n_audio, steps = 9, 7, 4, 6, 37, 6
patch = (1, 2, 2)
text = torch.randn(n_text, 40)
# A keyframe's vision block is tagged as video (0) inside the text rows.
text_tags = torch.tensor([1, 1, 0, 0, 0, 1, 1, 1, 1])
pos, tags, vidx, aidx, tidx, n_cond, n_cond_audio = MiniMaxH3PrepareLayoutStep.build_packed_sequence(
    text_tags, lt, dh, dw, n_audio, patch, 2, 2, 0, ("first", "last")
)
rows = torch.cat(
    [patchify_video_latents(torch.randn(1, 4, 1, dh, dw), patch) for _ in range(2)]
    + [patchify_video_latents(torch.randn(1, 4, lt, dh, dw), patch)]
)
audio = torch.randn(n_audio * 2, 6)
save("dit_text.bin", text.numpy())
save("dit_text_tags.bin", text_tags.float().numpy())
save("dit_video.bin", rows.numpy())
save("dit_audio.bin", audio.numpy())
save("dit_positions.bin", pos.float().numpy())
sv, sa = MiniMaxH3Scheduler(shift=12.0), MiniMaxH3Scheduler(shift=3.0)
sv.set_timesteps(steps)
sa.set_timesteps(steps)
save("sigmas_video.bin", sv.sigmas.numpy())
save("sigmas_audio.bin", sa.sigmas.numpy())
with torch.no_grad():
    for i, (t, ta) in enumerate(zip(sv.timesteps, sa.timesteps)):
        uniq, ridx = MiniMaxH3SetTimestepsStep.build_row_timesteps(
            vidx, aidx, n_cond, n_cond_audio, n_text, float(t), float(ta), max(float(t), 0.999), 1.0
        )
        v, a = dit(hidden_states=rows[None], audio_hidden_states=audio[None], encoder_hidden_states=text[None],
                   timestep=uniq, timestep_indices=ridx, token_tags=tags, position_ids=pos.float(),
                   video_indices=vidx, audio_indices=aidx, text_indices=tidx, return_dict=False)
        if i == 0:
            save("dit_v0.bin", v[0].numpy())
            save("dit_a0.bin", a[0].numpy())
        rows[n_cond:] = sv.step(v[0, n_cond:].float(), t, rows[n_cond:], return_dict=False)[0]
        audio = sa.step(a[0].float(), ta, audio, return_dict=False)[0]
save("dit_video_final.bin", rows.numpy())
save("dit_audio_final.bin", audio.numpy())
case.update({"lt": lt, "lh": dh, "lw": dw, "n_audio": n_audio, "steps": steps})
json.dump(case, open(os.path.join(out, "case.json"), "w"))
print("DiT: sequence", pos.shape[0], "rows,", len(sv.timesteps), "steps")
