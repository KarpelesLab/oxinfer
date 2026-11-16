#!/usr/bin/env python3
"""MiniMax-H3's audio VAE decoder (diffusers' AutoencoderKLMiniMaxH3Audio) on fixed audio rows,
for examples/h3_audio_check.rs.

usage: minimax_h3_audio_tiny.py OUT_DIR [AUDIO_VAE_DIR [LATENTS | signal]]
Without AUDIO_VAE_DIR a tiny random autoencoder is built and saved in OUT_DIR/audio_vae (three
stages upsampling by 5, 2 and 2 with kernels 9, 4 and 4); with it the release's `audio_vae/` is
loaded (float32). Writes OUT_DIR/audio_rows.bin (the stereo soundtrack as the DiT emits it:
`[2 * LATENTS, latent_channels]` normalized rows, channel-major) and OUT_DIR/audio_decoded.bin
(`[2, LATENTS * hop]` samples), little-endian f32. The rows are seeded normals, or with `signal`
the normalized posterior mode of diffusers' encoder on 5.175 s of a synthetic stereo signal (left:
a 100 Hz to 8 kHz sweep; right: a 220 Hz tone with four harmonics, amplitude-modulated at 3 Hz),
whose reconstruction error is printed. Then:
cargo run --release --example h3_audio_check -- OUT_DIR/audio_vae OUT_DIR    (or AUDIO_VAE_DIR OUT_DIR)

The rows go through what `MiniMaxH3VideoUnpackStep` / `MiniMaxH3AudioDecodeStep` do: one batch
item per channel, `latents * latents_std + latents_mean`, `audio_vae.decode`.
"""
import os, sys

import numpy as np
import torch
from diffusers import AutoencoderKLMiniMaxH3Audio

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
torch.manual_seed(0)

if len(sys.argv) > 2:
    vae = AutoencoderKLMiniMaxH3Audio.from_pretrained(sys.argv[2], torch_dtype=torch.float32).eval()
    signal = len(sys.argv) > 3 and sys.argv[3] == "signal"
    n = int(sys.argv[3]) if len(sys.argv) > 3 and not signal else 207  # a 124-frame request
else:
    C = 8
    vae = AutoencoderKLMiniMaxH3Audio(
        encoder_dim=4, encoder_rates=(2, 5, 2), latent_dim=32, latent_channels=C, num_attention_heads=2,
        decoder_dim=32, decoder_rates=(5, 2, 2), decoder_kernel_sizes=(9, 4, 4),
        latents_mean=[float(v) for v in torch.randn(C) * 0.3], latents_std=[float(v) for v in torch.rand(C) + 0.5],
    ).float().eval()
    with torch.no_grad():
        for name, p in vae.named_parameters():
            if name.endswith("weight_g"):
                p.copy_(torch.rand_like(p) * 0.5 + 0.25)
            elif name.endswith(("alpha", "beta")):
                p.copy_(torch.randn_like(p) * 0.5)
            else:
                p.copy_(torch.randn_like(p) * 0.3)
    vae.save_pretrained(os.path.join(out, "audio_vae"))
    n, signal = 37, False

C = vae.config.latent_channels
mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1)
std = torch.tensor(vae.config.latents_std).view(1, -1, 1)
if signal:
    sr = vae.config.sampling_rate
    t = torch.arange(n * vae.hop_length, dtype=torch.float64) / sr
    dur = t[-1] + 1 / sr
    sweep = 0.5 * torch.sin(2 * np.pi * 100 * dur / np.log(80) * (torch.exp(t / dur * np.log(80)) - 1))
    tone = sum(torch.sin(2 * np.pi * 220 * k * t) / k for k in range(1, 6)) * 0.25 * (0.6 + 0.4 * torch.sin(2 * np.pi * 3 * t))
    wave = torch.stack([sweep, tone]).float()[:, None]
    with torch.no_grad():
        z = vae.encode(wave, return_dict=False)[0].mode()
    rows = ((z - mean) / std).permute(0, 2, 1).reshape(2 * n, C).contiguous()
else:
    rows = torch.randn(2 * n, C)
latents = rows.reshape(2, n, C).permute(0, 2, 1).contiguous() * std + mean
with torch.no_grad():
    audio = vae.decode(latents, return_dict=False)[0].float()[:, 0]
rows.numpy().astype(np.float32).tofile(os.path.join(out, "audio_rows.bin"))
audio.numpy().astype(np.float32).tofile(os.path.join(out, "audio_decoded.bin"))
clipped = (audio.abs() >= 1.0).float().mean().item()
if signal:
    for name, a, b in zip(("left (sweep)", "right (tone)"), audio, wave[:, 0]):
        print(f"{name}: reconstruction SNR {10 * torch.log10(b.pow(2).mean() / (a - b).pow(2).mean()):.1f} dB, "
              f"correlation {torch.corrcoef(torch.stack([a, b]))[0, 1]:.4f}")
print(f"rows {tuple(rows.shape)} -> audio {tuple(audio.shape)} at {vae.config.sampling_rate} Hz, "
      f"mean |v| {audio.abs().mean():.4f}, {100 * clipped:.2f}% of the samples at the clamp")
