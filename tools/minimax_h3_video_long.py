#!/usr/bin/env python3
"""A tiny random MiniMax-H3 transformer (diffusers' MiniMaxH3Transformer3DModel) with the real model's
heads (128 wide, 16 rotary frequencies per axis) on the packed sequence of a real request, and its
outputs, for examples/h3_video_check.rs: what the tiny cases of minimax_h3_video_tiny.py (191 rows)
do not reach: thousands of rows, the positions of 37 latent frames, a text-to-video layout (no
keyframes) and a turbo LoRA's schedule (5 sigma points, shifts 6 / 3).

usage: minimax_h3_video_long.py OUT_DIR N_TEXT LT LH LW N_AUDIO
  448x256, 124 frames, a 124-token prompt (4682 rows):  OUT_DIR 124 37 16 28 207
environment: HEADS (3), HD (128), HIDDEN (96), LAYERS (3), SCALE (0.1: the weights' standard deviation)
Writes OUT_DIR/transformer (save_pretrained), the inputs, the first step's velocities, the rows
after the schedule and OUT_DIR/case.json (with the shifts and the absence of keyframes). Then:
cargo run --release --example h3_video_check -- OUT_DIR --transformer OUT_DIR/transformer
"""
import json, os, sys
import numpy as np, torch
from diffusers import MiniMaxH3Scheduler, MiniMaxH3Transformer3DModel
from diffusers.modular_pipelines.minimax_h3.before_denoise import (
    MiniMaxH3PrepareLayoutStep, MiniMaxH3SetTimestepsStep, patchify_video_latents)

out = sys.argv[1]
n_text, lt, dh, dw, n_audio = [int(v) for v in sys.argv[2:7]]
heads = int(os.environ.get("HEADS", 3)); hd = int(os.environ.get("HD", 128)); hidden = int(os.environ.get("HIDDEN", 96))
layers = int(os.environ.get("LAYERS", 3)); scale = float(os.environ.get("SCALE", 0.1))
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
C, AC, TD = 24, 32, 40
dit = MiniMaxH3Transformer3DModel(
    num_attention_heads=heads, attention_head_dim=hd, hidden_size=hidden, num_layers=layers, num_refiner_layers=1, ffn_dim=128,
    in_channels=C, audio_in_channels=AC, patch_size=(1, 2, 2), text_dim=TD, freq_dim=32, time_embed_hidden_dim=48,
    time_embed_dim=24, rope_freq_dim=16).float().eval()
randomize(dit, scale)
dit.save_pretrained(os.path.join(out, "transformer"))
steps = 5
patch = (1, 2, 2)
text = torch.randn(n_text, TD)
text_tags = torch.ones(n_text, dtype=torch.long)
pos, tags, vidx, aidx, tidx, n_cond, n_cond_audio = MiniMaxH3PrepareLayoutStep.build_packed_sequence(
    text_tags, lt, dh, dw, n_audio, patch, 2, 2, 0, ())
rows = patchify_video_latents(torch.randn(1, C, lt, dh, dw), patch)
audio = torch.randn(n_audio * 2, AC)
save("dit_text.bin", text.numpy()); save("dit_text_tags.bin", text_tags.float().numpy())
save("dit_video.bin", rows.numpy()); save("dit_audio.bin", audio.numpy()); save("dit_positions.bin", pos.float().numpy())
sv, sa = MiniMaxH3Scheduler(shift=6.0), MiniMaxH3Scheduler(shift=3.0)
sv.set_timesteps(steps); sa.set_timesteps(steps)
save("sigmas_video.bin", sv.sigmas.numpy()); save("sigmas_audio.bin", sa.sigmas.numpy())
print("sigmas", sv.sigmas, sa.sigmas)
with torch.no_grad():
    for i, (t, ta) in enumerate(zip(sv.timesteps, sa.timesteps)):
        uniq, ridx = MiniMaxH3SetTimestepsStep.build_row_timesteps(
            vidx, aidx, n_cond, n_cond_audio, n_text, float(t), float(ta), max(float(t), 0.999), 1.0)
        v, a = dit(hidden_states=rows[None], audio_hidden_states=audio[None], encoder_hidden_states=text[None],
                   timestep=uniq, timestep_indices=ridx, token_tags=tags, position_ids=pos.float(),
                   video_indices=vidx, audio_indices=aidx, text_indices=tidx, return_dict=False)
        if i == 0:
            save("dit_v0.bin", v[0].numpy()); save("dit_a0.bin", a[0].numpy())
        rows[n_cond:] = sv.step(v[0, n_cond:].float(), t, rows[n_cond:], return_dict=False)[0]
        audio = sa.step(a[0].float(), ta, audio, return_dict=False)[0]
save("dit_video_final.bin", rows.numpy()); save("dit_audio_final.bin", audio.numpy())
json.dump({"lt": lt, "lh": dh, "lw": dw, "n_audio": n_audio, "steps": steps, "shifts": [6.0, 3.0], "anchors": []},
          open(os.path.join(out, "case.json"), "w"))
print("DiT: sequence", pos.shape[0], "rows,", len(sv.timesteps), "steps")
